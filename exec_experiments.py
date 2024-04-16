import time
from optparse import OptionParser

import subprocess
import logging
import os
import sys


def remove_lines(filename):
    with open(filename, 'r') as fp:
        # read an store all lines into list
        lines = fp.readlines()

    # Write file
    with open(filename, 'w') as fp:
        # iterate each line
        for number, line in enumerate(lines):
            # delete line 5 and 8. or pass any Nth line you want to remove
            # note list index starts from 0
            if "[2m[36m(launch_and_fit pid" not in line and "[2m[36m(launch_and_get_parameters" not in line:
                fp.write(line)


logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ****** Experiments descriptions ******
# ======================================================================
# EXPERIMENT 1
# Algorithm = 'None'
# ======================================================================
# EXPERIMENT 2
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and  new_client = FALSE
# ======================================================================
# EXPERIMENT 3
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and new_client = TRUE new_clients_train = FALSE
# ======================================================================
# EXPERIMENT 4
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and new_client = TRUE new_clients_train = TRUE
# ======================================================================
# EXPERIMENT 5-----------------------------------------
# Algorithm = 'POC, poc=[0.1, 0.2, 0.3]' and new_client = TRUE new_clients_train = TRUE
#  Local epochs = 2
# ======================================================================
# Configurations
TYPE = 'torch'
# DATASETS      				= ['MNIST', 'CIFAR10', 'Tiny-ImageNet']
DATASETS = ['GTSRB']
# DATASETS      					= ['UCIHAR', 'MotionSense']
MODELS = ['CNN_1']
ALGORITHMS = ['None', 'POC', 'DEEV']
EPOCHS = {'1': [1], '2': [1], '3': [1], '4': [1], '5': [2], '6': [1], '7': [1], '8': [1], '9': [1], '10': [1],
          '11': [1], '12': [1], '13': [1], '14': [1], '15': [1], '16': [1], '17': [1], '18': [1], '19': [1], '20': [1],
          '21': [1], '22': [1], '23': [1], '24': [1], '25': [1], '26': [1], '27': [1], '28': [1], '29': [1], '30': [1],
          '31': [1], '32': [1], '40': [1], '41': [1], '42': [1], '43': [1]}
# CLIENTS       				= {'MNIST': 50, 'CIFAR10': 50, 'CIFAR100': 50, 'MotionSense': 50, 'UCIHAR': 50}
CLASSES = {'MNIST': 10, 'CIFAR10': 10, 'Tiny-ImageNet': 200, 'UCI-HAR': 6, 'EMNIST': 47, 'State Farm': 10, 'GTSRB': 43, 'ExtraSensory': 9, 'WISDM-WATCH': 12, 'WISDM-P': 12, 'Cologne': 12}
CLIENTS = {'MNIST': [8], 'CIFAR10': [20], 'EMNIST': [20], 'CIFAR100': [50], 'MotionSense': [24], 'UCI-HAR': [1],
           'Tiny-ImageNet': [2], 'State Farm': [10], 'GTSRB': [20], 'ExtraSensory': [1], 'WISDM-WATCH': [20], 'WISDM-P': [20], 'Cologne': [20]}
ALPHA = [0.1, 1.0]
# ALPHA = [1]
FRACTION_FIT = {'None': [0.3, 0.5, 0.7], 'POC': [0.3], 'DEEV': [], 'RAWCS': [0.3, 0.5, 0.7
                                                                   ]}
SPECIFIC_PARAMETERS = {'FedAVG': {'use_gradient': 'True', 'bits': 8}, 'FedKD': {'use_gradient': '', 'bits': 8},
                       'FedPAQ': {'use_gradient': 'True', 'bits': 8}, 'FedDistill': {'use_gradient': '', 'bits': 8},
                       'FedPredict': {'use_gradient': 'True', 'bits': 8}, 'FedPredict_Dynamic': {'use_gradient': 'True', 'bits': 8}, 'FedPer_with_FedPredict': {'use_gradient': 'True', 'bits': 8},
                       'FedPer': {'use_gradient': '', 'bits': 8}, 'FedAvgM': {'use_gradient': '', 'bits': 8},
                       'FedYogi': {'use_gradient': 'True', 'bits': 8}, 'FedProto': {'use_gradient': '', 'bits': 8}, 'FedClassAvg': {'use_gradient': '', 'bits': 8},
                       'FedYogi_with_FedPredict': {'use_gradient': 'True', 'bits': 8}, 'FedClustering': {'use_gradient': 'True', 'bits': 8},
                       'FedAla': {'use_gradient': '', 'bits': 8}, 'FedKD_with_FedPredict': {'use_gradient': '', 'bits': 8},
                       'FedSparsification': {'use_gradient': '', 'bits': 8}, 'CDA-FedAvg': {'use_gradient': '', 'bits': 8},
                       'FedCDM':  {'use_gradient': '', 'bits': 8}, 'CDA-FedAvg_with_FedPredict_Dynamic': {'use_gradient': '', 'bits': 8},
                       'CDA-FedAvg_with_FedPredict': {'use_gradient': '', 'bits': 8}, 'FedCDM_with_FedPredict_Dynamic': {'use_gradient': '', 'bits': 8},
                       'FedCDM_with_FedPredict':  {'use_gradient': '', 'bits': 8}}
POC = {'None': [], 'POC': [], 'DEEV': []}
DECAY = {'None': 0, 'POC': 0, 'DEEV': 0.001, 'RAWCS': 0}
NEW_CLIENTS = {'None': ['FALSE'], 'POC': ['FALSE', 'TRUE']}
NEW_CLIENTS_TRAIN = {'FALSE': ['FALSE'], 'TRUE': ['FALSE', 'TRUE']}
# DECAY         				= (0.001, 0.005, 0.009)
ROUNDS = 100
# STRATEGIES 					= ('FedPredict', 'FedPer', 'FedClassAvg', 'FedAVG', 'FedClassAvg_with_FedPredict', 'FedPer_with_FedPredict', 'FedProto', 'FedYogi', 'FedLocal',)
# STRATEGIES_FOR_ANALYSIS = ['FedKD', 'FedAVG', 'FedPAQ']
# STRATEGIES_TO_EXECUTE = ['FedKD', 'FedAVG']
STRATEGIES_FOR_ANALYSIS = {'40': ['FedAVG', 'FedPredict'], '43': ['FedAVG', 'FedPredict'], '41': ['FedPredict'],  '3': [], '6': ['FedPredict'], '7': ['FedPredict'], '14': ['FedSparsification'], '10': [], '11': [], '15': [], '16': ['FedAVG'], '17': ['FedPredict'], '19': ['FedPredict'], '22': ['FedPredict'], '26': ['FedPredict'], '30': ['FedPredict'], '31': ['FedPredict'], '32': ['FedPredict']}
STRATEGIES_TO_EXECUTE = {'40': ['FedAVG', 'FedPredict'], '43': ['FedAVG', 'FedPredict'], '41': ['FedPredict'], '3': ['FedClassAvg'],  '6': ['FedPredict_Dynamic'], '7': [], '10': ['FedPredict_Dynamic'], '11': ['FedPredict_Dynamic'], '14': ['FedAVG'], '15': ['FedClassAvg'], '16': ['FedAVG'], '17': ['FedPredict'], '19': ['FedPredict'], '22': ['FedPredict'], '26': ['FedPredict'], '30': ['FedPredict'], '31': ['FedPredict'], '32': ['FedPredict']}
N_CLUSTERS = [3]
CLUSTERING = "Yes"
CLUSTER_ROUND = [int(ROUNDS*0.5)]
CLUSTER_METRIC = ['weights']
CLUSTER_METHOD = ['KCenter']
LAYER_METRIC = [-1]

EXPERIMENTS = {
    1: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2, 'comment': '',
        'compression_method': 4, 'dynamic_data': "no"},
    2: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"},
    3: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'False',
        'class_per_client': 2,
        'comment': 'set', 'compression_method': "dls_compredict", 'dynamic_data': "no"},
    4: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'True',
        'class_per_client': 2,
        'comment': """apos a rodada {}, apenas novos clientes sao testados - novos clientes treinam apenas 1 vez (um round) - """.format(
            int(ROUNDS * 0.7)), 'compression_method': 4, 'dynamic_data': "no"},
    5: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'True',
        'class_per_client': 2,
        'comment': """apos a rodada {}, apenas novos clientes sao testados - novos clientes treinam apenas 1 vez (um round) com duas épocas locais """.format(
            int(ROUNDS * 0.7)), 'compression_method': 4, 'dynamic_data': "no"},
    6: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "dls", 'dynamic_data': "no"},
    7: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "compredict", 'dynamic_data': "no"},
    8: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2, 'comment': '',
        'compression_methods': 3, 'dynamic_data': "no"},
    9: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2, 'comment': '',
        'compression_methods': 4, 'dynamic_data': "no"},
    10: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "synthetic"},
    11:{'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "synthetic_global"},
    12: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'inverted', 'compression_methods': 3, 'dynamic_data': "no"},
    13: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'inverted', 'compression_methods': 4, 'dynamic_data': "no"},
    14: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "sparsification", 'dynamic_data': "no"},
    15: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"},
    16: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "dls", 'dynamic_data': "no"},
    17: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "per", 'dynamic_data': "no"},
    19: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "fedkd", 'dynamic_data': "no"},
    20: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 13, 'dynamic_data': "no"},
    21: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 14, 'dynamic_data': "no"},
    22: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 10, 'dynamic_data': "no"},
    23: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 24, 'dynamic_data': "no"},
    24: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 123, 'dynamic_data': "no"},
    25: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 50, 'dynamic_data': "no"},
    26: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 10, 'dynamic_data': "no"},
    27: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 234, 'dynamic_data': "no"},
    28: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 34, 'dynamic_data': "no"},
    29: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': 134, 'dynamic_data': "no"},
    30: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': -1, 'dynamic_data': "no"},
    31: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': -2, 'dynamic_data': "no"},
    32: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_methods': -3, 'dynamic_data': "no"},
    40: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"},
    41: {'algorithm': 'POC', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"},
    42: {'algorithm': 'DEEV', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"},
    43: {'algorithm': 'RAWCS', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
         'comment': 'set', 'compression_method': "no", 'dynamic_data': "no"}
    }


def execute_experiment(experiment, algorithm, new_client, new_client_train, comment, type, class_per_client,
                       compression, dynamic_data):
    if experiment == 9:
        comment = ROUNDS - 10
    elif experiment == 10:
        comment = ROUNDS + 10
    comment = str(comment)
    try:
        print("ola")
        clustering = ""
        for dataset in DATASETS:
            classes = CLASSES[dataset]
            for model in MODELS:
                for epochs in EPOCHS[experiment]:
                    for clients in CLIENTS[dataset]:
                        for fraction_fit in FRACTION_FIT[algorithm]:
                            for alpha in ALPHA:
                                decay = DECAY[algorithm]
                                for n_cluster in N_CLUSTERS:
                                    for cluster_round in CLUSTER_ROUND:
                                        for cluster_metric in CLUSTER_METRIC:
                                            for cluster_method in CLUSTER_METHOD:
                                                for layer_metric in LAYER_METRIC:
                                                    for strategy in STRATEGIES_TO_EXECUTE[experiment]:
                                                        use_gradient = SPECIFIC_PARAMETERS[strategy]['use_gradient']
                                                        poc = fraction_fit
                                                        # if strategy == 'FedPredict' and fraction_fit == 0.3 and new_client == 'False':
                                                        #     print("Pulou ", strategy, fraction_fit)
                                                        #     continue
                                                        # if int(experiment) == 2 and dataset == 'EMNIST':
                                                        #     print("conti")
                                                        #     continue
                                                        if "cluster" in strategy.lower():
                                                            clustering = "False"
                                                        else:
                                                            clustering = ""

                                                        print(
                                                            f'Starting {strategy} fraction_fit-{fraction_fit} simulation for {dataset} clients with {model} model ...',
                                                            os.getcwd())
                                                        test_config = """python {}/simulation.py --dataset='{}' --model='{}' --strategy='{}' --epochs={} --round={} --client={} --type='{}' --non-iid={} --aggregation_method='{}' --fraction_fit={} --poc={} --new_clients={} --new_clients_train={} --decay={} --comment={} --class_per_client={} --alpha={} --compression_method={} --classes={} --use_gradient={} --n_clusters={} --clustering={} --cluster_round={} --cluster_metric={} --metric_layer={} --cluster_method={} --dynamic_data={}""".format(
                                                            os.getcwd(), dataset, model,
                                                            strategy, epochs, ROUNDS, clients, TYPE, True, algorithm, fraction_fit, poc,
                                                            new_client, new_client_train, decay, comment, class_per_client, alpha,
                                                            compression, classes, use_gradient, n_cluster, clustering,
                                                            cluster_round, cluster_metric, layer_metric, cluster_method, dynamic_data)
                                                        print("=====================================\nExecutando... \n", test_config,
                                                              "\n=====================================")
                                                        # exit()
                                                        subprocess.Popen(test_config, shell=True).wait()
                                                        pass
                                                        # subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
                                                        # subprocess.Popen(['rm', '/tmp/*.py']).wait()
                                                    strategies_arg = ""
                                                    for i in STRATEGIES_FOR_ANALYSIS[experiment]:
                                                        strategies_arg = strategies_arg + """ --strategy='{}'""".format(i)
                                                    if len(STRATEGIES_FOR_ANALYSIS) > 0:
                                                        analytics_result_dir = """python analysis/non_iid.py --dataset='{}' --model='{}' --round={} --client={} --aggregation_method='{}' --poc={} --new_clients={} --new_clients_train={} --non-iid={} --comment='{}' --epochs={} --decay={} --fraction_fit={} --class_per_client={} --alpha={} --compression={}  --dynamic_data={} --experiment={} {}""".format(
                                                            dataset, model, ROUNDS, clients, algorithm, poc, new_client,
                                                            new_client_train,
                                                            True, comment, epochs, decay, fraction_fit, class_per_client, alpha,
                                                            compression, dynamic_data, experiment, strategies_arg)
                                                        print("=====================================\nExecutando analytics... \n",
                                                              analytics_result_dir,
                                                              "\n=====================================")
                                                        subprocess.Popen(analytics_result_dir, shell=True).wait()

                        # subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
                        # subprocess.Popen(['rm', '/tmp/*.py']).wait()
    except Exception as e:
        print("Error on execution")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def main():
    parser = OptionParser()

    # parser.add_option("-a", "--algorithm", dest="algorithm", default='None',   help="Algorithm used for selecting clients", metavar="STR")
    parser.add_option("", "--experiment_id", dest="experiment_id", default=1, help="", metavar="INT")
    parser.add_option("", "--type", dest="type", default="torch", help="", metavar="STR")

    (opt, args) = parser.parse_args()

    experiment = EXPERIMENTS[int(opt.experiment_id)]
    execute_experiment(experiment=opt.experiment_id, algorithm=experiment['algorithm'],
                       new_client=experiment['new_client'],
                       new_client_train=experiment['new_client_train'], comment=experiment['comment'], type=opt.type,
                       class_per_client=experiment['class_per_client'],
                       compression=experiment['compression_method'],
                       dynamic_data=experiment['dynamic_data'])
    # remove_lines("""execution_log/experiment_{}.txt""".format(opt.experiment_id))


if __name__ == '__main__':
    time.sleep(4)
    main()
