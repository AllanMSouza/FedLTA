import copy

import numpy as np
import pandas as pd
from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from pathlib import Path
import os
import sys
from matplotlib.lines import Line2D

class JointAnalysis():
    def __int__(self):
        pass

    def build_filenames_and_read(self, type, strategies, fractions_fit, datasets, experiments, alphas, rounds, clients='20', model='CNN', file_type='evaluate_client.csv'):

        self.rounds = rounds
        self.clients = clients
        self.fraction_fit = fractions_fit
        df_concat = None
        count = 0
        original_model = copy.deepcopy(model)
        version_dict = {"FedPredict": {-2: "FedPredict_{dc}"}}
        self.client_selections = ['Random', 'POC', 'RAWCS']
        # self.client_selections = ['Random', 'POC']
        for i in experiments:
            experiment = experiments[i]
            new_clients = experiment['new_client']
            new_clients_train = experiment['new_client_train']
            local_epochs = experiment['local_epochs']
            comment = experiment['comment']
            compression_list = [experiment['compression']]
            dynamic_data = experiment['dynamic_data']

            for dataset in datasets:

                for fraction_fit in fractions_fit:

                    for strategy in strategies:

                        for compression in compression_list:

                            if compression == 10 and "FedPredict" not in strategy:
                                continue

                            # if new_clients == 'True' and alpha == 5.0 and ("FedAVG" in strategy):
                            #     continue
                            # if strategy == "FedYogi_with_FedPredict" and (not(i == 2 and dataset == "CIFAR10") and not (i == 2 and fraction_fit in [0.2, 0.3] and dataset == "EMNIST")):
                            #     continue

                            for alpha in alphas:

                                if dataset in ['Cologne', 'WISDM-WATCH', 'WISDM-P']:
                                    model = "GRU"
                                else:
                                    model = original_model

                                for client_selection in self.client_selections:
                                    filename1 = """{}/{}/{}-{}-{}/new_clients_{}_train_{}_dynamic_data_{}/{}/{}/{}/{}/alpha_{}/{}_rounds/{}/{}_comment/{}_compression/""".format(os.path.abspath(os.path.join(os.getcwd(),
                                                                                                                            os.pardir)) + "/FL-H.IAAC/logs",
                                                                                                                            type,
                                                                                                                            strategy,
                                                                                                                            client_selection.replace("Random", "None"),
                                                                                                                            fraction_fit,
                                                                                                                            new_clients,
                                                                                                                            new_clients_train,
                                                                                                                            dynamic_data,
                                                                                                                            clients,
                                                                                                                            model,
                                                                                                                            dataset,
                                                                                                                            "classes_per_client_2",
                                                                                                                            alpha,
                                                                                                                            rounds,
                                                                                                                            local_epochs,
                                                                                                                            comment,
                                                                                                                            compression)

                                    server_filename = filename1 + "server.csv"
                                    filename1 = filename1 + file_type

                                    # print(filename1)
                                    try:
                                        flag = False
                                        # if Path(filename1).exists():
                                        df = pd.read_csv(filename1).dropna()
                                        df_server = pd.read_csv(server_filename)
                                        flag = True

                                        # elif Path(filename2).exists():
                                        #     df = pd.read_csv(filename2).dropna()
                                        #     flag = False

                                        if dataset == "GTSRB" and not flag and new_clients == 'train':
                                            print("nao achou: ", filename1)
                                        if dataset in ['EMNIST', 'CIFAR10'] and "CDA-FedAvg" in strategies:
                                            continue

                                    except:
                                        print("sem: ", filename1)
                                        continue
                                    if strategy == "FedPredict" and compression == "no":
                                        st = "FedAvg"
                                        s = "+FP"
                                    elif strategy == "FedCDM_with_FedPredict" and compression == "no":
                                        st = "FedCDM"
                                        s = "+FP"
                                    elif strategy == "CDA-FedAvg_with_FedPredict" and compression == "no":
                                        st = "CDA-FedAvg"
                                        s = "+FP"
                                    elif strategy == "FedPredict_Dynamic" and compression == "no":
                                        st = "FedAvg"
                                        s = r"+FP$_{DYN}$"
                                    elif strategy == "FedCDM_with_FedPredict_Dynamic" and compression == "no":
                                        st = "FedCDM"
                                        s = r"+FP$_{DYN}$"
                                    elif strategy == "CDA-FedAvg_with_FedPredict_Dynamic" and compression == "no":
                                        st = "CDA-FedAvg"
                                        s = r"+FP$_{DYN}$"
                                    elif strategy == "FedYogi_with_FedPredict" and compression == "no":
                                        st = "FedYogi"
                                        s = "+FP"
                                    elif strategy == "FedKD_with_FedPredict" and compression == "no":
                                        st = "FedKD"
                                        s = "+FP"
                                    else:
                                        st = copy.copy(strategy).replace('FedAVG', 'FedAvg')
                                        s = "Original"

                                    level = {0.3: 'Low', 0.5: 'Medium', 0.7: 'High'}[fraction_fit]

                                    print("f1: ", filename1)
                                    print(server_filename)
                                    print(df_server)
                                    # print("f2: ", filename2)
                                    if len(df) == 0:
                                        continue
                                    df['Strategy'] = np.array([st] * len(df))
                                    df['Version'] = np.array([s] * len(df))
                                    df['Experiment'] = np.array([i] * len(df))
                                    df['Fraction fit'] = np.array([fraction_fit] * len(df))
                                    df['Selection level'] = np.array([level] * len(df))
                                    df['Dataset'] = np.array([dataset.replace("WATCH", "W").replace("CIFAR10", "CIFAR-10")] * len(df))
                                    df['Solution'] = np.array([compression] * len(df))
                                    df['Alpha'] = np.array([alpha] * len(df))
                                    df['Client selection'] = np.array([client_selection.replace("None", "Random").replace('Random', 'Random')] * len(df))
                                    df['Method'] = np.array([st + s.replace("Original", "")] * len(df))
                                    df_server['Round'] = df_server['Server round'].to_numpy()
                                    if client_selection != 'RAWCS':
                                        df_server['Training cost'] = np.array([fraction_fit] * len(df_server))
                                    df_server = df_server[['Accuracy aggregated', 'Round', 'Training cost', 'Macro f1-score']].sort_values('Round')
                                    acc = df_server['Accuracy aggregated'].to_numpy()
                                    fscore = df_server['Macro f1-score'].to_numpy()
                                    t_c = df_server['Training cost'].to_numpy()
                                    ef = (acc * 100)/(t_c * 20)
                                    df_server['Eficiência'] = ef
                                    df = df.join(df_server, on='Round', how='inner', rsuffix='_2')
                                    if level == 'Low' and client_selection == 'RAWCS' and alpha == 0.1 and s == '+FP':
                                        print(server_filename)
                                        print("di: ", df_server)
                                    print(df)
                                    # else:
                                    #     df['Training cost'] = df['Fraction fit'].to_numpy()
                                    #     acc = df_server['Accuracy'].to_numpy()
                                    #     t_c = df['Training cost'].to_numpy()
                                    #     ef = (acc * 100) / (t_c * 20)
                                    #     df['Eficiência'] = ef

                                    # print(len([i for i in range(1, self.rounds + 1)]), len(eficiência), eficiência)
                                    # exit()
                                    # print(df)
                                    # exit()
                                    # print(eficiência)
                                    # exit()
                                    # efficiency = accuracy/(df['Training cost'].to_numpy()[0] * 20)
                                    if count == 0:
                                        df_concat = df
                                    else:
                                        df_concat = pd.concat([df_concat, df], ignore_index=True)

                                    count += 1

        # df_concat = self.convert_shared_layers(df_concat)
        print(df_concat)
        df_concat['Accuracy (%)'] = df_concat['Accuracy'] * 100
        df_concat['F1-score médio (%)'] = df_concat['Macro f1-score'].to_numpy() * 100
        df_concat['Round (t)'] = df_concat['Round']
        # plots
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        # self.joint_plot_acc_two_plots(df=df_concat, experiment=1, pocs=pocs)
        print("versao1: ", df_concat[['Strategy', 'Version']].drop_duplicates())
        print(df_concat[['Strategy', 'Fraction fit', 'Dataset', 'Version', 'Alpha', 'Client selection', 'Method']].drop_duplicates())
        # exit()
        df_concat = df_concat[['Round (t)', 'Loss', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Selection level', 'Experiment', 'Fraction fit', 'Dataset', 'Eficiência', 'Version', 'Alpha', 'F1-score médio (%)', 'Client selection', 'Method', 'Training cost']].groupby(['Round (t)', 'Strategy', 'Selection level', 'Experiment', 'Fraction fit', 'Dataset', 'Version', 'Alpha', 'Client selection', 'Method', 'Eficiência']).apply(lambda e: self.groupb_by_plot(e)).reset_index()[['Round (t)', 'Strategy', 'Selection level', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)', 'Loss', 'Version', 'Alpha', 'F1-score médio (%)', 'Client selection', 'Method', 'Eficiência', 'Training cost']]
        # df_concat = df_concat[df_concat['Client selection'] == 'RAWCS']
        # f = df_concat.query("Alpha==0.1 and Version=='+FP'")[['Round (t)', 'Selection level', 'Eficiência']].sort_values('Round (t)').drop_duplicates().to_numpy()
        # print(len(f), f)
        # exit()
        self.joint_plot_acc_four_plots(df_test=df_concat, experiment=40, alphas=alphas, dataset='CIFAR-10')
        self.joint_plot_acc_four_plots_efficiency(df_test=df_concat, experiment=40, alphas=alphas, dataset='CIFAR-10')
        # exit()
        self.joint_plot_acc_four_plots(df_test=df_concat, experiment=40, alphas=alphas, dataset='GTSRB')
        self.joint_plot_acc_four_plots_efficiency(df_test=df_concat, experiment=40, alphas=alphas, dataset='GTSRB')
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=2, alphas=alphas)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=3, alphas=alphas)
        # self.joint_plot_acc_four_plots(df=df_concat, experiment=4, fractions_fit=fractions_fit)
        # table




        print("estrategias: ", strategies)
        print("versao: ", df_concat[['Strategy', 'Version']].drop_duplicates())
        df_concat = self.convert(df_concat)
        strategies = df_concat['Strategy'].unique().tolist()
        print("unicas: ", strategies)
        print("versao2: ", df_concat['Strategy'].unique().tolist())
        aux = []
        # r'CDA-FedAvg+FP$_{DYN}$' 'CDA-FedAvg+FP',
        order = [r'FedAvg+FP$_{DYN}$', 'FedAvg+FP', 'FedAvg', r'FedCDM+FP$_{DYN}$', 'FedCDM+FP', 'FedCDM', r'CDA-FedAvg+FP$_{DYN}$', 'CDA-FedAvg+FP', 'CDA-FedAvg', 'FedPer']
        for s in order:
            if s in strategies:
                aux.append(s)
        strategies = aux
        print("finai: ", strategies)
        print("Experimento 1")
        metric_column = 'Accuracy (%)'
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='Low', metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='Low', metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='Medium', metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='Medium', metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='High', metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='High', metric_column=metric_column)
        metric_column = 'Eficiência'
        print("""calcular rawcs: {}""".format(df[
            df['Client selection'] == 'RAWCS']['Training cost'].mean()))
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='Low',
                         metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='Low',
                         metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='Medium',
                         metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='Medium',
                         metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='CIFAR-10', nivel='High',
                         metric_column=metric_column)
        self.joint_table(df_concat, alphas, strategies, experiment=40, dataset='GTSRB', nivel='High',
                         metric_column=metric_column)
        print("Experimento 2")
        # self.joint_table(df_concat, alphas, strategies, experiment=2)
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=3)
        # self.joint_table(df_concat, fractions_fit, strategies, experiment=4)

    def convert(self, df):

        versions = df['Version'].tolist()
        strategies = df['Strategy'].tolist()

        for i in range(len(versions)):
            version = versions[i]
            strategy = strategies[i]

            if version == r"+FP$_{DYN}$":
                # o = strategy
                strategy = "" + strategy + r"+FP$_{DYN}$"
                strategies[i] = strategy
                # if o == 'CDA-FedAvg':
                #     print("entr", strategy)
                #     exit()
            elif version == "+FP":
                strategy = "" + strategy + "+FP"
                strategies[i] = strategy
            else:
                strategy = "" + strategy + ""
                strategies[i] = strategy

        print(np.unique(strategies))
        # exit()
        df['Strategy'] = np.array(strategies)

        return df

    def convert_shared_layers(self, df):

        shared_layers_list = df['Solution'].tolist()
        strategies = df['Strategy'].tolist()
        for i in range(len(shared_layers_list)):
            shared_layer = str(shared_layers_list[i])
            strategy = strategies[i]
            if "FedPredict" in strategy:
                if "dls" in shared_layer:
                    shared_layers_list[i] = "FedPredict_{d}"
                    continue
                elif "dls_compredict" in shared_layer:
                    shared_layers_list[i] = "FedPredict_{dc}"
                    continue
                elif "compredict" in shared_layer:
                    shared_layers_list[i] = "FedPredict_{c}"
                    continue
                elif "no" in shared_layer:
                    shared_layers_list[i] = "FedPredict"
                    continue
            else:
                shared_layers_list[i] = strategy

        df['Strategy'] = np.array(shared_layers_list)

        return df

    def groupb_by_table(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = df['Accuracy (%)'].mean()

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy]})

    def accuracy_improvement(self, df):

        df_difference = copy.deepcopy(df)
        columns = df.columns.tolist()
        indexes = df.index.tolist()

        # datasets = ['Cologne', 'WISDM-P', 'WISDM-WATCH']
        datasets = ['Random', 'POC', 'RAWCS']
        solutions = pd.Series([i[1] for i in indexes]).unique().tolist()
        reference_solutions = {}
        for solution_key in solutions:
            if r"FP$_{DYN}$" in solution_key:
                reference_solutions[solution_key] = solution_key.replace(r"+FP$_{DYN}$", "")
            elif "FP" in solution_key:
                reference_solutions[solution_key] = solution_key.replace("+FP", "")

        # print("indexes: ", indexes)
        # print("reference: ", reference_solutions)

        for dataset in datasets:
            for solution in reference_solutions:
                reference_index = (dataset, solution)
                target_index = (dataset, reference_solutions[solution])

                for column in columns:
                    print((df.loc[reference_index, column][:4]), reference_index, column)
                    difference = str(round(float(df.loc[reference_index, column][:4].replace(u"\u00B1", "")) - float(
                        df.loc[target_index, column][:4].replace(u"\u00B1", "")), 1))
                    difference = str(
                        round(float(difference) * 100 / float(df.loc[target_index, column][:4].replace(u"\u00B1", "")),
                              1))
                    if difference[0] != "-":
                        difference = "textuparrow" + difference
                    else:
                        difference = "textdownarrow" + difference.replace("-", "")
                    df_difference.loc[reference_index, column] = "(" + difference + "%)" + df.loc[reference_index, column]


        # print(indexes)
        # print(indexes[0])
        # print(df_difference)

        return df_difference


    def joint_table(self, df, pocs, strategies, experiment, dataset, nivel, metric_column):

        df = df[df['Selection level'] == nivel]

        alpha = 0.1
        print("""calcular rawcs {} {} {}: f {}""".format(dataset, nivel, alpha, df.query("Alpha=="+str(alpha))[df['Client selection'] == 'RAWCS']['Training cost'].mean()))
        alpha = 1.0
        print("""calcular rawcs {} {} {}: f {}""".format(dataset, nivel, alpha, df.query("Alpha==" + str(alpha))[
            df['Client selection'] == 'RAWCS']['Training cost'].mean()))

        model_report = {i: {} for i in df['Alpha'].unique().tolist()}
        if experiment == 1:
            df = df[df['Round (t)'].isin(range(70, 81))]
        # df_test = df[['Round (t)', 'Size of parameters', 'Strategy', 'Accuracy (%)', 'Experiment', 'Fraction fit', 'Dataset']].groupby(
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset']).apply(
        #     lambda e: self.groupb_by_table(e)).reset_index()[
        #     ['Round (t)', 'Strategy', 'Experiment', 'Fraction fit', 'Dataset', 'Size of parameters (bytes)', 'Accuracy (%)']]
        elif experiment == 2:
            df = df[df['Round (t)'].isin(range(70, 81))]
            pass
        df_test = df[
            ['Round (t)', 'Size of parameters (bytes)', 'Strategy', 'Accuracy (%)', 'Experiment', 'Client selection', 'Fraction fit', 'Dataset', 'Alpha', 'Eficiência']]

        # df_test = df_test.query("""Round in [10, 100]""")
        print("agrupou table")
        print(df_test)
        convert_dict = {0.1: 5, 0.2: 10, 0.3: 15, 0.5: 20}
        # df_test['Fraction fit'] = np.array([convert_dict[i] for i in df_test['Fraction fit'].tolist()])

        columns = strategies

        # index = [np.array(['Cologne'] * len(columns) + ['WISDM-W'] * len(columns) + ['WISDM-P'] * len(columns)), np.array(columns * 3)]
        index = [np.array( ['Random'] * len(columns) + ['POC'] * len(columns) + ['RAWCS'] * len(columns)),
                 np.array(columns * 3)]

        models_dict = {}
        ci = 0.95
        print("filtro tabela")
        fraction_fit = 0.3
        client_selection = 'Random'
        for model_name in model_report:

            cologne_acc = {}
            mnist_acc = {}
            cifar10_acc = {}
            for column in columns:

                # mnist_acc[column] = (self.filter(df_test, experiment, 'MNIST', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cifar10_acc[column] = (self.filter(df_test, experiment, 'CIFAR10', float(column), strategy=model_name)['Accuracy (%)']*100).mean().round(6)
                # cologne_acc[column] = self.t_distribution((self.filter(df_test, experiment, 'Cologne',
                #                                                        float(model_name), strategy=column)[
                #     'Accuracy (%)']).tolist(), ci)
                mnist_acc[column] = self.t_distribution((self.filter(df=df_test, experiment=experiment, dataset=dataset,
                                                                     client_selection='Random', fraction_fit=fraction_fit, alpha=model_name, strategy=column)[
                                         metric_column]).tolist(), ci)
                cifar10_acc[column] = self.t_distribution((self.filter(df=df_test, experiment=experiment, dataset=dataset,
                                                                     client_selection='POC',
                                                                     fraction_fit=fraction_fit, alpha=model_name,
                                                                     strategy=column)[
                    metric_column]).tolist(), ci)
                cologne_acc[column] = self.t_distribution(
                    (self.filter(df=df_test, experiment=experiment, dataset=dataset,
                                 client_selection='RAWCS',
                                 fraction_fit=fraction_fit, alpha=model_name,
                                 strategy=column)[
                        metric_column]).tolist(), ci)
                # gtsrb_acc[column] = self.t_distribution(
                #     (self.filter(df_test, experiment, 'GTSRB', float(model_name), strategy=column)[
                #         'Accuracy (%)']).tolist(), ci)

            model_metrics = []

            # for column in columns:
            #     print("pegou: ", len(cologne_acc[column]), type(cologne_acc[column]))
            #     model_metrics.append(cologne_acc[column])
            for column in columns:
                print("pegou: ", len(mnist_acc[column]), type(mnist_acc[column]))
                model_metrics.append(mnist_acc[column])
            for column in columns:
                print("pegou 2: ", len(mnist_acc[column]), type(mnist_acc[column]))
                model_metrics.append(cifar10_acc[column])
            for column in columns:
                print("pegou 2: ", len(mnist_acc[column]), type(mnist_acc[column]))
                model_metrics.append(cologne_acc[column])

            models_dict[model_name] = model_metrics

        for key in models_dict:

            print("""chave {} tamanho {} ti {}""".format(key, len(models_dict[key]), type(models_dict)))

        print("indice: ", len(index))
        df_table = pd.DataFrame(models_dict, index=index).round(4)
        print("df table: ", df_table)
        print(df_table.to_string())

        df_accuracy_improvements = self.accuracy_improvement(df_table)

        indexes = df_table.index.tolist()
        n_solutions = len(pd.Series([i[1] for i in indexes]).unique().tolist())
        max_values = self.idmax(df_table, n_solutions)
        print("max values", max_values)

        for max_value in max_values:
            row_index = max_value[0]
            column = max_value[1]
            column_values = df_accuracy_improvements[column].tolist()
            column_values[row_index] = "textbf{" + str(column_values[row_index]) + "}"

            df_accuracy_improvements[column] = np.array(column_values)

        df_accuracy_improvements.columns = np.array(list(model_report.keys()))
        print("melhorias")
        print(df_accuracy_improvements)

        indexes = [0.1, 1.0]
        for i in range(df_accuracy_improvements.shape[0]):
            row = df_accuracy_improvements.iloc[i]
            for index in indexes:
                value_string = row[index]
                add_textbf = False
                if "textbf{" in value_string:
                    value_string = value_string.replace("textbf{", "").replace("}", "")
                    add_textbf = True

                if ")" in value_string:
                    value_string = value_string.replace("(", "").split(")")
                    gain = value_string[0]
                    acc = value_string[1]
                else:
                    gain = ""
                    acc = value_string

                if add_textbf:
                    if gain != "":
                        gain = "textbf{" + gain + "}"
                    acc = "textbf{" + acc + "}"

                row[index] = acc + " & " + gain

            df_accuracy_improvements.iloc[i] = row

        latex = df_accuracy_improvements.to_latex().replace("\\\nEMNIST", "\\\n\hline\nEMNIST").replace("\\\nGTSRB", "\\\n\hline\nGTSRB").replace("\\\nCIFAR-10", "\\\n\hline\nCIFAR-10").replace("\\bottomrule", "\\hline\n\\bottomrule").replace("\\midrule", "\\hline\n\\midrule").replace("\\toprule", "\\hline\n\\toprule").replace("textbf", r"\textbf").replace("\}", "}").replace("\{", "{").replace("\\begin{tabular", "\\resizebox{\columnwidth}{!}{\\begin{tabular}").replace("\$", "$").replace("textuparrow", r"\textuparrow").replace("textdownarrow", r"\textdownarrow").replace("\&", "&").replace("&  &", "& - &").replace("\_", "_").replace("&  \\", "& - \\").replace(" - " + r"\textbf", " " + r"\textbf").replace("_{DYN}", r"$_{\text{DYN}}$").replace("WISDM-W", r"\parbox[t]{2mm}{\multirow{10}{*}{\rotatebox[origin=c]{90}{WISDM-W}}}").replace("WISDM-P", r"\parbox[t]{2mm}{\multirow{10}{*}{\rotatebox[origin=c]{90}{WISDM-P}}}").replace("Random", " & Random").replace("POC", fr"{nivel} & POC").replace("& FedAvg &", "& & FedAvg &").replace("RAWCS", "& RAWCS")

        base_dir = """analysis/output/experiment_{}/alphas_{}/datasets_{}/client_selection/""".format(str(experiment + 1), [0.1, 1.0], ['CIFAR-10', 'GTSRB'])
        filename = """{}latex_{}_{}_{}_{}.txt""".format(base_dir, str(experiment), dataset, nivel, metric_column)
        pd.DataFrame({'latex': [latex]}).to_csv(filename, header=False, index=False)

        self.improvements(df_table, experiment + 1, metric_column)

        #  df.to_latex().replace("\}", "}").replace("\{", "{").replace("\\\nRecall", "\\\n\hline\nRecall").replace("\\\nF-score", "\\\n\hline\nF1-score")



    def improvements(self, df, experiment, metric_column):

        # r"CDA-FedAvg+FP$_{DYN}$": "CDA-FedAvg", r"CDA-FedAvg+FP": "CDA-FedAvg"
        strategies = {r"FedAvg+FP": "FedAvg"}
        datasets = ['Random', 'POC', 'RAWCS']
        print(df)
        # exit()
        columns = df.columns.tolist()
        improvements_dict = {'Dataset': [], 'Strategy': [], 'Original strategy': [], 'Alpha': [], metric_column: []}
        df_improvements = pd.DataFrame(improvements_dict)

        for dataset in datasets:
            for strategy in strategies:
                original_strategy = strategies[strategy]

                for j in range(len(columns)):

                    index = (dataset, strategy)
                    index_original = (dataset, original_strategy)
                    print(index)
                    print(df.loc[index])
                    acc = float(df.loc[index].tolist()[j].replace("textbf{", "")[:4].replace(u"\u00B1", ""))
                    acc_original = float(df.loc[index_original].tolist()[j].replace("textbf{", "")[:4].replace(u"\u00B1", ""))

                    row = {'Dataset': dataset, 'Strategy': strategy, 'Original strategy': original_strategy, 'Alpha': columns[j], metric_column: acc - acc_original}

                    df_improvements = df_improvements.append(row, ignore_index=True)

        print("Experiment: ", experiment + 1)
        print(df_improvements)

    def groupb_by_plot(self, df):
        parameters = int(df['Size of parameters'].mean())
        accuracy = float(df['Accuracy (%)'].mean())
        f1_score = float(df['F1-score médio (%)'].mean())
        loss = float(df['Loss'].mean())
        if df['Client selection'].to_numpy()[0] == 'DEEV':
            print("deev: ", df['Training cost'].to_numpy())
        accuracy = np.round(accuracy, 2)
        t_c = df['Training cost'].mean()
        # print(df['Training cost'].to_numpy()[0] * 20, accuracy, accuracy/(df['Training cost'].to_numpy()[0] * 20))
        # exit()

        return pd.DataFrame({'Size of parameters (bytes)': [parameters], 'Accuracy (%)': [accuracy],
                             'F1-score médio (%)': [f1_score], 'Loss': [loss], 'Training cost': [t_c]})

    def filter(self, df, experiment, dataset, client_selection, fraction_fit, alpha, strategy=None):

        # df['Accuracy (%)'] = df['Accuracy (%)']*100
        if strategy is not None:
            query = """Experiment=={} and Dataset=='{}' and Strategy=='{}'""".format(str(experiment), str(dataset), strategy)
            df = df.query(query)
            df = df[df['Alpha'] == alpha]
        else:
            query = """Experiment=={} and Dataset=='{}'""".format(str(experiment), (dataset))
            df = df.query(query)
            df = df[df['Alpha'] == alpha]

        print("antesr55: ", df, client_selection)
        df = df[df['Client selection'] == client_selection]
        # df = df[df['Fraction fit'] == fraction_fit]

        print("query: ", query, client_selection, fraction_fit)
        print("filtrou: ", df, experiment, dataset, alpha, strategy)

        return df

    def filter_and_plot(self, ax, base_dir, filename, title, df, experiment, dataset, alpha, x_column, y_column, client_selection, fraction_fit, hue, hue_order=None, style=None, markers=None, size=None, sizes=None, y_max=1, y_lim=True, style_order=None):

        print("antes: ", df)
        df = self.filter(df, experiment, dataset, client_selection, fraction_fit, alpha)
        print("ttt: ", df)
        df['Strategy'] = np.array(["" + i + "" for i in df['Strategy'].tolist()])

        print("filtrado: ", df, df[hue].unique().tolist())
        line_plot(df=df, base_dir=base_dir, file_name=filename, x_column=x_column, y_column=y_column, title=title, hue=hue, ax=ax, tipo='' + str(experiment), hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes, y_max=y_max, y_lim=y_lim, style_order=style_order)

    def joint_plot_acc_four_plots(self, df_test, experiment, alphas, dataset):
        print("Joint plot exeprimento: ", experiment)

        datast = df_test['Dataset'].unique().tolist()
        print("agrupou plot")
        print(df_test[df_test['Round (t)']==20])
        print(df_test)
        # exit()
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        rows = len(alphas)
        cols = len(datast) + 1
        fig, axs = plt.subplots(rows, cols,  sharex='all', sharey='all', figsize=(9, 6))

        x_column = 'Round (t)'
        y_column = 'Accuracy (%)'
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        base_dir = """analysis/output/experiment_{}/alphas_{}/datasets_{}/client_selection/""".format(str(experiment+1), alphas, datast)
        # ====================================================================
        for i in range(rows):
            for j in range(cols):
                alpha = alphas[i]
                # dataset = datast[j]
                # dataset = 'GTSRB'
                # dataset = 'CIFAR-10'
                client_selection = ['Random', 'POC', 'RAWCS'][j]
                # client_selection = 'RAWCS'
                fraction_fit = self.fraction_fit[j]
                print("cf: ", client_selection, fraction_fit)
                title = """\u03B1={}; {}""".format(alpha, client_selection)
                filename = ''
                hue_order = ['FedAvg+FP', 'FedAvg']
                # hue_order = ['FedAvg', 'CDA-FedAvg', 'FedCDM', 'FedPer']
                # hue_order = None
                style = 'Selection level'
                # "+FP",
                # style_order = [r"+FP$_{DYN}$",  "+FP", "Original"]
                style_order = ['High', 'Medium', 'Low']
                y_max = 100
                # markers = [',', '.'
                markers = None
                size = None
                # sizes = (1, 1.8)
                sizes = None
                self.filter_and_plot(ax=axs[i,j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                                     experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column, client_selection=client_selection, fraction_fit=fraction_fit,
                                     hue='Method', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes, y_max=y_max, style_order=style_order)
                # if i != 1 and j != 0:
                #     axs[i,j].get_legend().remove()
                #     axs[i,j].legend(fontsize=7)

                axs[i,j].set_xlabel('')
                axs[i,j].set_ylabel('')

        axs[0, 0].get_legend().remove()
        axs[0, 1].get_legend().remove()
        axs[1, 0].get_legend().remove()
        axs[1, 1].get_legend().remove()
        axs[1, 2].get_legend().remove()
        axs[0, 2].get_legend().remove()

        # =========================///////////================================
        fig.suptitle(dataset, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        # plt.subplots_adjust(right=0.9)
        # fig.legend(
        #            loc="Lower right")
        # fig.legend(lines, labels)
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        fig.supxlabel(x_column, y=-0.02)
        fig.supylabel(y_column, x=-0.01)

        if experiment == 1:
            axs[0, 0].annotate('1', xy=(73, 80), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                        xytext=(87, 88),
                        arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[0, 0].annotate('2', xy=(68, 51), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(44, 48),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[0, 1].annotate('3', xy=(68, 20), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(50, 30),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
        elif experiment == 2:
            axs[0, 1].annotate('5', xy=(86, 32), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(84, 48),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[1, 0].annotate('4', xy=(68, 78), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(42, 76),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))


        lines_labels = [axs[0, 0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        print("linhas")
        print(lines)
        print(lines[0].get_color(), lines[0].get_ls())
        print("rotulos")
        print(labels)
        # # exit()
        colors = []
        markers = []
        for i in range(len(lines)):
            color = lines[i].get_color()
            colors.append(color)
            ls = lines[i].get_ls()
            if ls not in ["o"]:
                ls = "o"
        markers = ["", "-", "--", "dotted"]

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
        print(handles)
        print("---")
        print(labels)
        if dataset != "CIFAR-10":
            axs[1, 1].legend(handles, labels, fontsize=7)
        else:
            axs[1, 2].legend(handles, labels, fontsize=7)
        print("base: ", base_dir, """{}joint_plot_four_plot_{}_{}_rounds_{}_clients.png""".format(base_dir, str(experiment), self.rounds, self.clients))
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients_{}.png""".format(base_dir, str(experiment), self.rounds, self.clients, dataset), bbox_inches='tight', dpi=400)
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients_{}.svg""".format(base_dir, str(experiment), self.rounds, self.clients, dataset), bbox_inches='tight', dpi=400)
        fig.savefig(
            """{}joint_plot_four_plot_{}_{}_rounds_{}_clients_{}.pdf""".format(base_dir, str(experiment), self.rounds,
                                                                               self.clients, dataset),
            bbox_inches='tight', dpi=400)

    def joint_plot_acc_four_plots_efficiency(self, df_test, experiment, alphas, dataset):
        print("Joint plot exeprimento: ", experiment)

        datast = df_test['Dataset'].unique().tolist()
        print("agrupou plot")
        print(df_test[df_test['Round (t)']==20])
        print(df_test)
        # exit()
        # figsize=(12, 9),
        sns.set(style='whitegrid')
        rows = len(alphas)
        cols = len(datast) + 1
        fractions_fit = [0.3]
        fig, axs = plt.subplots(rows, cols,  sharex='all', sharey='all', figsize=(9, 6))

        x_column = 'Round (t)'
        y_column = 'Eficiência'
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        base_dir = """analysis/output/experiment_{}/alphas_{}/datasets_{}/client_selection/""".format(str(experiment+1), alphas, datast)
        # ====================================================================
        for i in range(rows):
            for j in range(cols):
                alpha = alphas[i]
                # dataset = datast[j]
                # dataset = 'GTSRB'
                # dataset = 'CIFAR-10'
                client_selection = ['Random', 'POC', 'RAWCS'][j]
                # client_selection = 'RAWCS'
                fraction_fit = self.fraction_fit[j]
                print("cf: ", client_selection, fraction_fit)
                title = """\u03B1={}; {}""".format(alpha, client_selection)
                filename = ''
                hue_order = ['FedAvg+FP', 'FedAvg']
                # hue_order = ['FedAvg', 'CDA-FedAvg', 'FedCDM', 'FedPer']
                # hue_order = None
                style = 'Selection level'
                # "+FP",
                # style_order = [r"+FP$_{DYN}$",  "+FP", "Original"]
                style_order = ['High', 'Medium', 'Low']
                y_max = 45
                # markers = [',', '.'
                markers = None
                size = None
                # sizes = (1, 1.8)
                sizes = None
                self.filter_and_plot(ax=axs[i,j], base_dir=base_dir, filename=filename, title=title, df=df_test,
                                     experiment=experiment, dataset=dataset, alpha=alpha, x_column=x_column, y_column=y_column, client_selection=client_selection, fraction_fit=fraction_fit,
                                     hue='Method', hue_order=hue_order, style=style, markers=markers, size=size, sizes=sizes, y_max=y_max, style_order=style_order)
                # if i != 1 and j != 0:
                #     axs[i,j].get_legend().remove()
                #     axs[i,j].legend(fontsize=7)

                axs[i,j].set_xlabel('')
                axs[i,j].set_ylabel('')

        axs[0, 0].get_legend().remove()
        axs[0, 1].get_legend().remove()
        axs[1, 0].get_legend().remove()
        axs[1, 1].get_legend().remove()
        axs[1, 2].get_legend().remove()
        axs[0, 2].get_legend().remove()
        # axs[1, 1].legend(fontsize=5)

        # =========================///////////================================
        fig.suptitle(dataset, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, hspace=0.14)
        # plt.subplots_adjust(right=0.9)
        # fig.legend(
        #            loc="Lower right")
        # fig.legend(lines, labels)
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        fig.supxlabel(x_column, y=-0.02)
        fig.supylabel(y_column, x=-0.01)

        if experiment == 1:
            axs[0, 0].annotate('1', xy=(73, 80), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                        xytext=(87, 88),
                        arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[0, 0].annotate('2', xy=(68, 51), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(44, 48),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[0, 1].annotate('3', xy=(68, 20), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(50, 30),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
        elif experiment == 2:
            axs[0, 1].annotate('5', xy=(86, 32), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(84, 48),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))
            axs[1, 0].annotate('4', xy=(68, 78), xycoords='data', bbox=dict(boxstyle="circle", fc="w", color='black'),
                               xytext=(42, 76),
                               arrowprops=dict(width=1, headwidth=4, facecolor='black', color='black'))


        lines_labels = [axs[0, 0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        print("linhas")
        print(lines)
        print(lines[0].get_color(), lines[0].get_ls())
        print("rotulos")
        print(labels)
        # # exit()
        colors = []
        markers = []
        for i in range(len(lines)):
            color = lines[i].get_color()
            colors.append(color)
            ls = lines[i].get_ls()
            if ls not in ["o"]:
                ls = "o"
        markers = ["", "-", "--", "dotted"]

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(len(markers))]
        print(handles)
        print("---")
        print(labels)
        axs[1, 1].legend(handles, labels, fontsize=7)
        print("base: ", base_dir, """{}joint_plot_four_plot_{}_{}_rounds_{}_clients.png""".format(base_dir, str(experiment), self.rounds, self.clients))
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients_efficiency_{}.png""".format(base_dir, str(experiment), self.rounds, self.clients, dataset), bbox_inches='tight', dpi=400)
        fig.savefig("""{}joint_plot_four_plot_{}_{}_rounds_{}_clients_efficiency_{}.svg""".format(base_dir, str(experiment), self.rounds, self.clients, dataset), bbox_inches='tight', dpi=400)
        fig.savefig(
            """{}joint_plot_four_plot_{}_{}_rounds_{}_clients_efficiency_{}.pdf""".format(base_dir, str(experiment), self.rounds,
                                                                               self.clients, dataset),
            bbox_inches='tight', dpi=400)

    def idmax(self, df, n_solutions):

        df_indexes = []
        columns = df.columns.tolist()
        print("colunas", columns)
        for i in range(len(columns)):

            column = columns[i]
            column_values = df[column].tolist()
            print("ddd", column_values)
            indexes = self.select_mean(i, column_values, columns, n_solutions)
            df_indexes += indexes

        return df_indexes

    def select_mean(self, index, column_values, columns, n_solutions):

        list_of_means = []
        indexes = []
        print("ola: ", column_values, "ola0")

        for i in range(len(column_values)):

            print("valor: ", column_values[i])
            value = float(str(str(column_values[i])[:4]).replace(u"\u00B1", ""))
            interval = float(str(column_values[i])[5:8])
            minimum = value - interval
            maximum = value + interval
            list_of_means.append((value, minimum, maximum))


        for i in range(0, len(list_of_means), n_solutions):

            dataset_values = list_of_means[i: i+n_solutions]
            max_tuple = max(dataset_values, key=lambda e: e[0])
            column_min_value = max_tuple[1]
            column_max_value = max_tuple[2]
            print("maximo: ", column_max_value)
            for j in range(len(list_of_means)):
                value_tuple = list_of_means[j]
                min_value = value_tuple[1]
                max_value = value_tuple[2]
                if j >= i and j < i+n_solutions:
                    if not(max_value < column_min_value or min_value > column_max_value):
                        indexes.append([j, columns[index]])

        return indexes

    def t_distribution(self, data, ci):

        min_ = st.t.interval(alpha=ci, df=len(data) - 1,
                      loc=np.mean(data),
                      scale=st.sem(data))[0]

        mean = np.mean(data)
        average_variation = (mean - min_).round(1)
        mean = mean.round(1)

        return str(mean) + u"\u00B1" + str(average_variation)

if __name__ == '__main__':
    """
        This code generates a joint plot (multiples plots in one image) and a table of accuracy.
        It is done for each experiment.
    """

    # 1: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
    #     'comment': 'set', 'compression': 'no', 'local_epochs': '1_local_epochs', 'dynamic_data': "synthetic"},
    experiments = {
        40: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
                    'comment': 'set', 'compression': 'no', 'local_epochs': '1_local_epochs', 'dynamic_data': "no"},
        41: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
             'comment': 'set', 'compression': 'no', 'local_epochs': '1_local_epochs', 'dynamic_data': "no"},
        42: {'algorithm': 'RAWCS', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
             'comment': 'set', 'compression': 'no', 'local_epochs': '1_local_epochs', 'dynamic_data': "no"}
    }

    # experiments = {
    #     1: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'class_per_client': 2,
    #         'comment': 'set', 'compression': 'no', 'local_epochs': '1_local_epochs', 'dynamic_data': "synthetic"}
    # }

    strategies = ['FedAVG', 'FedPredict']
    # 'FedPredict', 'FedYogi_with_FedPredict', 'FedKD_with_FedPredict', 'FedAVG', 'FedYogi', 'FedPer', 'FedProto', 'FedKD'
    # pocs = [0.1, 0.2, 0.3]
    fractions_fit = [0.3, 0.5, 0.7]
    # datasets = ['MNIST', 'CIFAR10']
    # datasets = ['CIFAR10']
    datasets = ['CIFAR10', 'GTSRB']
    alpha = [0.1, 1.0]
    rounds = 100
    clients = '20'
    model = 'CNN_1'
    type_t = 'torch'
    file_type = 'evaluate_client.csv'

    joint_plot = JointAnalysis()
    joint_plot.build_filenames_and_read(type_t, strategies, fractions_fit, datasets, experiments, alpha, rounds, clients, model, file_type)