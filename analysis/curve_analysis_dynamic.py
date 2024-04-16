from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from base_plots import bar_plot, line_plot
from pathlib import Path
import os
import ast

class Verify:

    def start(self):

        self.base_dir = "output/exp_decaying/"

        if not Path(self.base_dir).exists():
            os.makedirs(self.base_dir+"csv/")
            os.makedirs(self.base_dir + "png/")
            os.makedirs(self.base_dir + "svg/")

        self.server_analysis(0)
        # self.server_analysis(1)

    def curve(self, server_round, rounds_without_fit, start_round, sm):
        # 0
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # # normalizar dentro de 0 e 1
        # rounds_without_fit  = pow(min(rounds_without_fit, max_rounds_without_fit)/max_rounds_without_fit, alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        # 	# o denominador faz com que a curva se prolongue com menor decaimento
        # 	# Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        # 	eq1 = (-rounds_without_fit-(server_round)/beta)
        # 	# eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        # 	# do modelo global
        # 	eq2 = pow(2.7, eq1)
        # 	eq3 = min(eq2, 1)
        # 	global_model_weight = eq3
        # 1
        # max_rounds_without_fit = 3
        # alpha = 2
        # beta = 9
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(min(rounds_without_fit + 0.00001, max_rounds_without_fit) / (max_rounds_without_fit + 0.00001), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (- rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 3
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # start_round = 0
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 4
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # start_round = 5
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 5
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 12
        # start_round = 0
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 6
        # max_rounds_without_fit = 4
        # alpha = 1.2
        # beta = 9
        # # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
        # delta = 0.02
        # # normalizar dentro de 0 e 1
        # fx_rounds_without_fit = pow(
        #     min(rounds_without_fit + delta, max_rounds_without_fit), -alpha)
        # # global_model_weight = 1
        # # o denominador faz com que a curva se prolongue com menor decaimento
        # # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        # eq1 = (-fx_rounds_without_fit - ((rounds_without_fit) / beta))
        # # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        # # do modelo global
        # eq2 = np.exp(eq1)
        # eq3 = min(eq2, 1)
        # global_model_weight = eq3
        # 7
        # sigma = 5
        # mu = 10
        # eq1 = - pow((rounds_without_fit-mu), 2)/(1*sigma*sigma)
        # # eq2 = 1/(sigma*pow((2*np.pi), 1/2))
        # eq2 = 1/2
        # eq3 = eq2*np.exp(eq1)
        # global_model_weight = eq3
        # 8 plotar
        # max_rounds_without_fit = 100
        # alpha = 1.2
        # # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
        # delta = 0.01
        # # normalizar dentro de 0 e 1
        # updated_level = pow(
        #     min(rounds_without_fit + delta, max_rounds_without_fit), -alpha)
        # evolutionary_level = (server_round / 50)
        #
        # eq1 = (-updated_level - evolutionary_level)
        # eq2 = round(np.exp(eq1), 6)
        # global_model_weight = eq2
        #
        # # if rounds_without_fit == 0:
        # #     print(global_model_weight)
        #
        # return global_model_weight, updated_level
        # 9
        if rounds_without_fit == 0:
            global_model_weight = 0
            updated_level = 1
        else:
            # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
            # normalizar dentro de 0 e 1
            updated_level = 1 / rounds_without_fit
            evolutionary_level = (server_round / 100)

            eq1 = ((-updated_level - evolutionary_level)*sm)
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

            # if rounds_without_fit == 0:
            #     print(global_model_weight)

        return global_model_weight, updated_level

    def server_analysis(self, index):

        print("indice: ", index)
        rounds = 100
        rounds_without_fit = [1, 2, 5, 10, 100]
        rounds_without_fit_list = []
        start_round = 0
        x = []
        for i in range(1, rounds + 1):
            x.append(i)
        # x = [i for i in range(1, rounds + 1)]
        # rounds_without_fit_list = rounds_without_fit_list + [rounds_without_fit] * len(x)
        sm = [1, 0.5]
        sm_list = []
        y = []
        x_new = []
        for j in sm:
            for rounds_w_f in rounds_without_fit:
                y += [self.curve(i, rounds_w_f, start_round, j)[index] for i in x]
                rounds_without_fit_list = rounds_without_fit_list + [rounds_w_f] * len(x)
                sm_list = sm_list + [j] * len(x)
                x_new += x

        print("tamanho y: ", len(y), len(rounds_without_fit_list), len(x_new), len(sm_list))
        x_column = 'Round (t)'
        if index == 0:
            y_column = '$F(el, ul, s)$'
        else:
            y_column = 'ul'
        hue = 'nt'
        style = 's'
        hue_order = [100, 10, 5, 2, 1]
        print(len(x), len(y), len(rounds_without_fit_list))
        df = pd.DataFrame({x_column: x_new, y_column: y, hue: rounds_without_fit_list, style: sm_list})
        title = ""
        print("x: ", df[x_column].unique().tolist())
        df = df.drop_duplicates()
        # print(df.to_string())
        print("base dir: ", self.base_dir, index)
        # if index == 1:
        #     print(df.drop_duplicates(subset=[y_column, hue]))
        #     print("Ola")
        #     df = df.round(4)
        #     bar_plot(df=df,
        #               base_dir=self.base_dir,
        #               file_name="bar_als" + str(index),
        #               x_column=hue,
        #               y_column=y_column,
        #               title=title)
        # else:
        # fig, ax = plt.subplots(1,1)
        plt.plot()
        line_plot(df=df,
                 base_dir=self.base_dir,
                 file_name="curve_dynamic",
                 x_column=x_column,
                 y_column=y_column,
                 title=title,
                  ax=None,
                  hue=hue,
                  style=style,
                  y_lim=True,
                  y_max=1,
                  y_min=0,
                  hue_order=hue_order,
                  tipo='curve_dynamic')

        # lines_labels = [ax.get_legend_handles_labels()]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # colors = []
        # ax.get_legend().remove()
        # for i in range(len(lines)):
        #     color = lines[i].get_color()
        #     colors.append(color)
        #     ls = lines[i].get_ls()
        #     if ls not in ["o"]:
        #         ls = "o"
        # markers = ["", "-", "--"]
        # plt.grid()
        #
        # f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        # handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        # handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
        # ax.legend(handles, labels, fontsize=8, ncols=5)
        # fig.savefig("""{}/png/curve_als.png""".format(self.base_dir), bbox_inches='tight', dpi=400)
        # fig.savefig("""{}/svg/curve_als.svg""".format(self.base_dir,), bbox_inches='tight', dpi=400)

if __name__ == '__main__':

    Verify().start()