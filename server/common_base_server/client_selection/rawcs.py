import copy
import flwr as fl
import numpy as np
import math
import os
import sys
import time
import csv
import random

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from models.torch import DNN, Logistic, CNN, MobileNet, resnet20, CNN_EMNIST, MobileNetV2, CNN_X, CNN_5, CNN_2, CNN_3

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import csv
import json
import os
import pickle
import random
from typing import Optional, List, Tuple, Union, Dict

import flwr as fl
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, FitIns, FitRes, Scalar, parameters_to_ndarrays, EvaluateIns, \
    EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

def get_energy_by_completion_time(comp, comm, avg_joules):
    # Understanding Operational 5G: A First Measurement Study on Its Coverage, Performance and Energy Consumption
    # 55% of 4W (avg 5g watts in the energy consumption graphs)
    # 1W = 1J/s
    # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
    AVG_5G_JOULES = 2.22
    comm_joules = AVG_5G_JOULES * comm

    comp_joules = avg_joules * comp

    return comp_joules + comm_joules
def update_utility(energy, utility):
    utility = utility - energy
    return utility
def init_battery_level(max_battery):
    perc = random.randint(30, 100)
    utility = battery = max_battery * (perc / 100)

    return battery, utility
def idle_power_deduction(battery, elapsed_time):
    # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
    # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
    IDLE_CONSUMPTION = 0.1
    consumption = IDLE_CONSUMPTION * elapsed_time

    battery -= consumption

    return battery
def get_devices_battery_profiles(num_clients):
    # Common full mAh batteries
    # Values from commom smartphones specifications
    batteries_mah = random.choices([3500, 4000, 4500, 5000], k=num_clients)

    # Converting to joules
    # https://www.axconnectorlubricant.com/rce/battery-electronics-101.html#faq6
    # V = 3.85, commom value from specifications
    batteries_joules = [mAh * 3.85 * 3.6 for mAh in batteries_mah]

    # from: A Novel Non-invasive Method to Measure Power Consumption on Smartphones (39.27 mA = 0.15 W)
    # from: What can Android mobile app developers do about the energy consumption of machine learning (0.1 W)
    IDLE_CONSUMPTION = 0.1

    # Machine Learning at Facebook: Understanding Inference at the Edge
    # Energy Consumption of Batch and Online Data Stream Learning Models for Smartphone-based Human Activity Recognition
    min_battery = 2.0
    max_battery = 5.0
    battery_interval = max_battery - min_battery

    percentuals = [random.random() for _ in range(num_clients)]
    # 1W = 1J/s
    # from: https://www.stouchlighting.com/blog/electricity-and-energy-terms-in-led-lighting-j-kw-kwh-lm/w
    avg_joules = [IDLE_CONSUMPTION + min_battery + percentual * battery_interval for percentual in percentuals]


    return list(zip(batteries_joules, avg_joules))

class Rawcs(fl.server.strategy.FedAvg):
    
    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 type,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='',
                 non_iid=False,
                 model_name='',
                 new_clients=False,
                 new_clients_train=False
                 ):
        super().__init__(
            fraction_fit=float(args.fraction_fit), min_available_clients=num_clients,
            min_fit_clients=num_clients, min_evaluate_clients=num_clients
        )

        self.dataset_name = dataset
        self.num_clients = num_clients
        self.num_classes = n_classes
        self.eval_fraction = 1
        # self.min_fit = min_fit
        # self.min_eval = min_eval
        # self.min_avail = min_avail
        # self.results_dir = results_dir
        self.transmission_threshold = 0.2
        # self.devices_profile = devices_profile
        if fraction_fit == 0.3:
            battery = 0.45 # 0.5
            self.link_quality_lower_lim = 0.5  # lq_min
            self.limit_relationship_max_latency = 0  # pt_max
            level = 'low'
        elif fraction_fit == 0.5:
            # self.link_quality_lower_lim = 0.3  # lq_min
            # self.limit_relationship_max_latency = 0.3  # pt_max
            # level = 'medium'
            battery = 0.35 # 0.45
            self.link_quality_lower_lim = 0.05  # lq_min
            self.limit_relationship_max_latency = 0.4  # pt_max
            level = 'medium'
        elif fraction_fit == 0.7:
            battery = 0.05
            self.link_quality_lower_lim = 0.01  # lq_min
            self.limit_relationship_max_latency = 7 # pt_max
            level = 'high'
            # args.dataset.lower(),
        self.network_profiles = """./clients_selection_configuration_files/rawcs/sim_1_num_clients_{}_num_rounds_100.pkl""".format(num_clients)
        self.devices_profile = """./clients_selection_configuration_files/rawcs/profiles_sim_cifar10_seed_1_level_{}_alpha_{}_battery_{}.json""".format( level, args.alpha, battery)
        # self.devices_profile = """./clients_selection_configuration_files/rawcs/profiles_sim_Cifar10_seed_1.json"""
        # self.sim_idx = sim_idx
        # self.input_shape = input_shape
        self.battery_weight = 0.33
        self.cpu_cost_weight = 0.33
        self.link_prob_weight = 0.33
        self.target_accuracy = 1.0
        self.time_percentile = 95
        self.comp_latency_lim = np.inf
        self.clients_info = {}
        self.clients_last_round = []
        self.max_training_latency = 0.0

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        network_profiles = None

        with open(self.network_profiles, 'rb') as file:
            network_profiles = pickle.load(file)

        clients_training_time = []

        with open(self.devices_profile, 'r') as file:
            json_dict = json.load(file)
        # for key in range(self.num_clients):
        #     self.clients_info[key] = {}
        for key in [i for i in range(self.num_clients)]:
            self.clients_info[int(key)] = json_dict[str(key)]
            self.clients_info[int(key)]['perc_budget_10'] = False
            self.clients_info[int(key)]['perc_budget_20'] = False
            self.clients_info[int(key)]['perc_budget_30'] = False
            self.clients_info[int(key)]['perc_budget_40'] = False
            self.clients_info[int(key)]['perc_budget_50'] = False
            self.clients_info[int(key)]['perc_budget_60'] = False
            self.clients_info[int(key)]['perc_budget_70'] = False
            self.clients_info[int(key)]['perc_budget_80'] = False
            self.clients_info[int(key)]['perc_budget_90'] = False
            self.clients_info[int(key)]['perc_budget_100'] = False
            self.clients_info[int(key)]['initial_battery'] = self.clients_info[int(key)]['battery']
            if self.clients_info[int(key)]['total_train_latency'] > self.max_training_latency:
                self.max_training_latency = self.clients_info[int(key)]['total_train_latency']
            clients_training_time.append(self.clients_info[int(key)]['total_train_latency'])
            self.clients_info[int(key)]['network_profile'] = network_profiles[int(key)]

        self.comp_latency_lim = np.percentile(clients_training_time, self.time_percentile)

        self.limit_relationship_max_latency = self.comp_latency_lim / self.max_training_latency

        #client_manager.wait_for(self.num_clients)

        return super().initialize_parameters(client_manager)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[
        Tuple[ClientProxy, FitIns]]:

        if server_round == 1:
            selected_cids = [*range(0, self.num_clients, 1)]
        else:
            selected_cids = self.sample_fit()

        print("pre selected: ", selected_cids)

        selected_cids = self.filter_clients_to_train_by_predicted_behavior(selected_cids, server_round)

        # Return client/config pairs
        config = {
            "round": server_round,
            "selected_clients": ' '.join([str(cid) for cid in selected_cids]),
            "strategy": "rawcs-sp"
        }

        fit_ins = FitIns(parameters, config)

        # return [(client, fit_ins) for client in client_manager.sample(client_manager.num_available())]

        return selected_cids

    # def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    #     """Evaluate global model parameters using an evaluation function."""
    #     # Let's assume we won't perform the global model evaluation on the server side.
    #     return None
    #
    # def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
    #     """Return sample size and required number of strategy."""
    #     num_clients = int(num_available_clients * self.fraction_fit)
    #     return max(num_clients, self.min_fit), self.min_avail
    #
    # def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
    #     """Use a fraction of available strategy for evaluation."""
    #     num_clients = int(num_available_clients * self.eval_fraction)
    #     return max(num_clients, self.min_eval), self.min_avail
    #
    # def weighted_accuracy_avg(self, results: List[Tuple[int, float]]) -> float:
    #     """Aggregate evaluation results obtained from multiple strategy."""
    #     num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    #     weighted_acc = [num_examples * acc for num_examples, acc in results]
    #     return sum(weighted_acc) / num_total_evaluation_examples
    #
    # def weighted_matthews_ceof_avg(self, results: List[Tuple[int, float]]) -> float:
    #     """Aggregate evaluation results obtained from multiple strategy."""
    #     num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    #     weighted_mcc = [num_examples * mcc for num_examples, mcc in results]
    #     return sum(weighted_mcc) / num_total_evaluation_examples

    def sample_fit(self):
        selected_cids = []
        clients_with_resources = []

        for client in self.clients_info:
            if self.has_client_resources(client):
                clients_with_resources.append((client, self.clients_info[client]['accuracy']))

                client_cost = self.get_cost(self.battery_weight, self.cpu_cost_weight, self.link_prob_weight,
                                            self.clients_info[client]['battery'] /
                                            self.clients_info[client]['max_battery'],
                                            self.clients_info[client][
                                                'total_train_latency'] / self.max_training_latency,
                                            self.clients_info[client]['trans_prob'],
                                            self.target_accuracy,
                                            self.clients_info[client]['accuracy'])

                client_benefit = self.get_benefit()

                if random.random() <= (1 - client_cost / client_benefit):
                    selected_cids.append(client)

        if len(selected_cids) == 0 and len(clients_with_resources) != 0:
            clients_with_resources.sort(key=lambda client: client[1])
            selected_cids = [client[0] for client in clients_with_resources[:round(len(clients_with_resources)
                                                                                   * self.fraction_fit)]]
        if len(selected_cids) == 0:
            clients_with_battery = []

            for client in self.clients_info:
                if self.clients_info[client]['battery'] - \
                        self.clients_info[client]['delta_train_battery'] >= self.clients_info[client]['min_battery']:
                    clients_with_battery.append((client, self.clients_info[client]['accuracy']))

            clients_with_battery.sort(key=lambda client: client[1])

            selected_cids = [client[0] for client in
                             clients_with_battery[:round(len(clients_with_battery) * self.fraction_fit)]]

        return selected_cids

    def sample_eval(self, client_manager: ClientManager, num_clients):
        return client_manager.sample(num_clients)

    def filter_clients_to_train_by_predicted_behavior(self, selected_cids, server_round):
        # 1 - Atualiza tempo máximo de processamento
        # 2 - Atualiza bateria consumida pelos clientes em treinamento
        # 3 - Atualiza métricas: energia consumida, desperdício de energia, clientes dreanados, latência total
        # 4 - Identificar clientes que falharão a transmissão devido a alguma instabilidade
        # 5 - Atualizar métrica de consumo desperdiçado por falha do cliente
        # 6 - Atualiza lista de clientes que não completaram o treino por falta de bateria ou instabilidade da rede
        total_train_latency_round = 0.0
        total_energy_consumed = 0.0
        total_wasted_energy = 0.0
        round_depleted_battery_by_train = 0
        round_depleted_battery_total = 0
        round_transpassed_min_battery_level = 0
        max_latency = 0.0
        filtered_by_transmisssion = 0
        clients_to_not_train = []

        for cid in selected_cids:
            comp_latency = self.clients_info[cid]['train_latency']
            comm_latency = self.clients_info[cid]['comm_latency']
            avg_joules = self.clients_info[cid]["avg_joules"]

            client_latency = self.clients_info[cid]["total_train_latency"]
            client_consumed_energy = get_energy_by_completion_time(comp_latency, comm_latency, avg_joules)
            new_battery_value = self.clients_info[cid]['battery'] - client_consumed_energy

            if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_value < \
                    self.clients_info[cid]['min_battery']:
                round_transpassed_min_battery_level += 1

            if new_battery_value < 0:
                total_energy_consumed += self.clients_info[cid]['battery']
                total_wasted_energy += self.clients_info[cid]['battery']
                self.clients_info[cid]['battery'] = 0
                round_depleted_battery_by_train += 1
                round_depleted_battery_total += 1
                clients_to_not_train.append(cid)
            else:
                total_energy_consumed += client_consumed_energy
                self.clients_info[cid]['battery'] = new_battery_value

                if self.clients_info[cid]['network_profile'][server_round - 1] < self.transmission_threshold:
                    clients_to_not_train.append(cid)
                    total_wasted_energy += client_consumed_energy
                    filtered_by_transmisssion += 1
                else:
                    total_train_latency_round += client_latency

            if client_latency > max_latency and cid not in clients_to_not_train:
                max_latency = client_latency
        # 7 - Remove de clientes selecionados os que foram drenados pelo treinamento
        print("den: ", clients_to_not_train)
        filtered_selected_cids = list(set(selected_cids).difference(clients_to_not_train))
        # 8 - Calcular consumo em estado de espera
        # 9 - Atualizar bateria de cada cliente
        # 10 - Atualizar clientes que foram drenados sem que seja pelo treino
        for cid in self.clients_info:
            old_battery_level = self.clients_info[cid]['battery']

            if old_battery_level > 0:
                if cid not in filtered_selected_cids:
                    new_battery_level = idle_power_deduction(old_battery_level, max_latency)
                else:
                    idle_time = max_latency - (self.clients_info[cid]['total_train_latency'])
                    new_battery_level = idle_power_deduction(old_battery_level, idle_time)

                if self.clients_info[cid]['battery'] >= self.clients_info[cid]['min_battery'] and new_battery_level < \
                        self.clients_info[cid]['min_battery']:
                    round_transpassed_min_battery_level += 1

                if new_battery_level <= 0:
                    self.clients_info[cid]['battery'] = 0
                    round_depleted_battery_total += 1
                else:
                    self.clients_info[cid]['battery'] = new_battery_level

        perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, avg_battery_perc, batteries_perc = \
            self.transpassed_budget()

        # filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
        #     self.sim_idx) + f"_system_metrics_frac_{self.fraction_fit}_weights_{self.battery_weight}_{self.cpu_cost_weight}_{self.link_prob_weight}.csv"
        #
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        #
        # with open(filename, 'a') as log:
        #     log.write(f"{server_round},{total_train_latency_round},{total_energy_consumed},{total_wasted_energy},"
        #               f"{len(selected_cids)},{round_depleted_battery_by_train},{round_depleted_battery_total},"
        #               f"{filtered_by_transmisssion},{len(filtered_selected_cids)},"
        #               f"{round_transpassed_min_battery_level},{perc_budget_10},{perc_budget_20},{perc_budget_30},"
        #               f"{perc_budget_40},{perc_budget_50},{perc_budget_60},{perc_budget_70},{perc_budget_80},"
        #               f"{perc_budget_90},{perc_budget_100},{avg_battery_perc}\n"
        #               )
        #
        # filename = self.results_dir + self.__repr__() + "_" + self.sim_id + "_" + str(
        #     self.sim_idx) + f"_batteries_per_client_frac_{self.fraction_fit}_weights_{self.battery_weight}_{self.cpu_cost_weight}_{self.link_prob_weight}.csv"
        #
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        #
        # with open(filename, 'a') as log:
        #     write = csv.writer(log)
        #     write.writerow(batteries_perc)

        return [str(i) for i in filtered_selected_cids]

    def update_sample(self, client_manager, selected_cids):
        selected_clients = []

        for cid in selected_cids:
            selected_clients.append(client_manager.clients[str(cid)])

        return selected_clients

    def has_client_resources(self, client_id: int):
        if (self.clients_info[client_id]['battery'] - self.clients_info[client_id]['delta_train_battery']) \
                >= self.clients_info[client_id]['min_battery'] and self.clients_info[client_id]['trans_prob'] \
                >= self.link_quality_lower_lim and self.clients_info[client_id][
            'total_train_latency'] / self.max_training_latency <= self.limit_relationship_max_latency:
            return True

        return False

    def get_cost(self, battery_w: float, cpu_w: float, link_w: float, battery: float, cpu_relation: float,
                 link_qlt: float, target_accuracy: float, client_accuracy: float):
        return (((battery_w) * (1 - battery)) + ((cpu_w) * (cpu_relation)) + ((link_w) * (1 - link_qlt))) ** (
                target_accuracy - client_accuracy)

    def get_benefit(self):
        return self.target_accuracy

    def transpassed_budget(self):
        battery_perc = []
        perc_budget_10 = 0
        perc_budget_20 = 0
        perc_budget_30 = 0
        perc_budget_40 = 0
        perc_budget_50 = 0
        perc_budget_60 = 0
        perc_budget_70 = 0
        perc_budget_80 = 0
        perc_budget_90 = 0
        perc_budget_100 = 0
        for cid in self.clients_info:
            depletion = 1 - self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery']
            battery_perc.append(self.clients_info[cid]['battery'] / self.clients_info[cid]['initial_battery'])

            if not self.clients_info[cid]['perc_budget_10'] and depletion > 0.1:
                self.clients_info[cid]['perc_budget_10'] = True
            if not self.clients_info[cid]['perc_budget_20'] and depletion > 0.2:
                self.clients_info[cid]['perc_budget_20'] = True
            if not self.clients_info[cid]['perc_budget_30'] and depletion > 0.3:
                self.clients_info[cid]['perc_budget_30'] = True
            if not self.clients_info[cid]['perc_budget_40'] and depletion > 0.4:
                self.clients_info[cid]['perc_budget_40'] = True
            if not self.clients_info[cid]['perc_budget_50'] and depletion > 0.5:
                self.clients_info[cid]['perc_budget_50'] = True
            if not self.clients_info[cid]['perc_budget_60'] and depletion > 0.6:
                self.clients_info[cid]['perc_budget_60'] = True
            if not self.clients_info[cid]['perc_budget_70'] and depletion > 0.7:
                self.clients_info[cid]['perc_budget_70'] = True
            if not self.clients_info[cid]['perc_budget_80'] and depletion > 0.8:
                self.clients_info[cid]['perc_budget_80'] = True
            if not self.clients_info[cid]['perc_budget_90'] and depletion > 0.9:
                self.clients_info[cid]['perc_budget_90'] = True
            if not self.clients_info[cid]['perc_budget_100'] and depletion == 1.0:
                self.clients_info[cid]['perc_budget_100'] = True

            if self.clients_info[cid]['perc_budget_10']:
                perc_budget_10 += 1
            if self.clients_info[cid]['perc_budget_20']:
                perc_budget_20 += 1
            if self.clients_info[cid]['perc_budget_30']:
                perc_budget_30 += 1
            if self.clients_info[cid]['perc_budget_40']:
                perc_budget_40 += 1
            if self.clients_info[cid]['perc_budget_50']:
                perc_budget_50 += 1
            if self.clients_info[cid]['perc_budget_60']:
                perc_budget_60 += 1
            if self.clients_info[cid]['perc_budget_70']:
                perc_budget_70 += 1
            if self.clients_info[cid]['perc_budget_80']:
                perc_budget_80 += 1
            if self.clients_info[cid]['perc_budget_90']:
                perc_budget_90 += 1
            if self.clients_info[cid]['perc_budget_100']:
                perc_budget_100 += 1
        return perc_budget_10, perc_budget_100, perc_budget_20, perc_budget_30, perc_budget_40, perc_budget_50, \
            perc_budget_60, perc_budget_70, perc_budget_80, perc_budget_90, sum(battery_perc) / len(battery_perc), \
            battery_perc