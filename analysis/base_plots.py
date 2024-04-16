from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import numpy as np

# sns.color_palette()

def bar_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, hue_order=None, y_lim=False, y_min=0, y_max=100, log_scale=False, sci=False, x_order=None, ax=None, tipo=None, palette=None):
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    max_value = df[y_column].max()

    sns.set(style='whitegrid')
    log = ""
    file_name = """{}_barplot""".format(file_name)
    # df[y_column] = df[y_column].round(2)
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if sci:
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        # formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        # ax.set_ylim([0, 130000])
    if y_lim:
        # y_max = float(max_value)
        plt.ylim([y_min, y_max])


    # errorbar=('ci', 0.95),
    figure = sns.barplot(ax=ax, x=x_column, y=y_column, hue=hue, data=df, hue_order=hue_order,  order=x_order, palette=palette)
    if tipo == "classes":
        for bars in figure.containers:
            figure.bar_label(bars, fmt='%.f', padding=13, fontsize=9)
        figure.set_ylim(top=115)
    if tipo == "balance":
        for bars in figure.containers:
            figure.bar_label(bars, fmt='%.f', padding=12, fontsize=9)
        figure.set_ylim(top=115)
    if tipo == "nt":
        for bars in figure.containers:
            figure.bar_label(bars, fmt='%.f', padding=9, fontsize=9)
        figure.set_ylim(top=119)
    figure.set_title(title)

    if ax is None:
        fig, ax = plt.subplots()
        figure = figure.get_figure()
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def box_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, tipo=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None, x_order=None, palette=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    file_name = """{}_lineplot""".format(file_name)
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # if x_column is not None:
    #     x = df[x_column].tolist()
    #     plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if tipo is not None:
        # palette = sns.color_palette()
        figure = sns.boxplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, order=x_order)
    else:
        figure = sns.boxplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, order=x_order)

    figure.set_title(title)

    # if hue is not None:
    #     medians = df.groupby([x_column, hue])[y_column].median()
    # else:
    #     medians = df.groupby([x_column])[y_column].median()
    # print("mediana: ", medians)
    # medians = medians.round(3)
    vertical_offset = df[y_column].median() * 0.05  # offset from median for display

    # for xtick in figure.get_xticks():
    #     figure.text(xtick, medians[xtick] + vertical_offset, medians[xtick],
    #                   horizontalalignment='center', size='x-small', color='w', weight='semibold')

    if tipo == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if tipo == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def violin_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None, x_order=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    file_name = """{}_violinplot""".format(file_name)
    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # if x_column is not None:
    #     x = df[x_column].tolist()
    #     plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if type is not None:
        palette = sns.color_palette()
        figure = sns.violinplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, order=x_order).set_title(title)
    else:
        figure = sns.violinplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, order=x_order).set_title(title)

    if type == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)


def line_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, tipo=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None, markers=None, size=None, sizes=None, style_order=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    file_name = """{}_lineplot""".format(file_name)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    # sns.set(style='whitegrid')
    # ax.grid(False)
    if tipo is not None:
        if 'dynamic' in tipo and ax is not None:
            ax.grid(False)
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    x = df[x_column].tolist()
    # plt.xticks(np.arange(0, max(x) + 1, 2.0))
    print("titulo: ", title)
    if tipo is not None:
        palette = sns.color_palette()
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, style=style, style_order=style_order, markers=markers, size=size, sizes=sizes, linewidth=0.8).set_title(title)
    else:
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, style=style, style_order=style_order, markers=markers, size=size, sizes=sizes, linewidth=0.8).set_title(title)
    print("nof")

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if tipo == 2:
        pass
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
    #     plt.xticks(np.arange(0, max(x)+1, 50))
    #     plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if tipo == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    if tipo == 'dls':
        aa = figure.get_figure().axes[0]
        # print(ax)
        # exit()
        lines_labels = [aa.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        colors = []
        aa.get_legend().remove()
        for i in range(len(lines)):
            color = lines[i].get_color()
            colors.append(color)
            ls = lines[i].get_ls()
            if ls not in ["o"]:
                ls = "o"
        markers = ["", "-", "--"]
        # plt.grid()
        new_labels = []
        for i in range(len(labels)):
            if i != n:
                new_labels.append(labels[i])
            else:
                print("label: ", labels[i])
        new_labels[-1] = 'df=' + new_labels[-1]
        new_labels[-2] = 'df=' + new_labels[-2]

        new_labels_2 = []

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
        new_handles = []
        for i in range(len(handles)):
            if i not in [0, 6]:
                new_handles.append(handles[i])
                if i <= 6:
                    new_labels[i] = "nt=" + new_labels[i]
                new_labels_2.append(new_labels[i])
        aa.legend(new_handles, new_labels_2, fontsize=8, ncols=3)
    elif tipo == 'curve_dynamic':
        aa = figure.get_figure().axes[0]
        # print(ax)
        # exit()
        lines_labels = [aa.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        colors = []
        aa.get_legend().remove()
        for i in range(len(lines)):
            color = lines[i].get_color()
            colors.append(color)
            ls = lines[i].get_ls()
            if ls not in ["o"]:
                ls = "o"
        markers = ["", "-", "--"]
        # plt.grid()
        new_labels = []
        for i in range(len(labels)):
            if i != n:
                new_labels.append(labels[i])
            else:
                print("label: ", labels[i])
        new_labels[-1] = 's=' + new_labels[-1]
        new_labels[-2] = 's=' + new_labels[-2]

        new_labels_2 = []

        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f("o", colors[i]) for i in range(len(hue_order) + 1)]
        handles += [plt.Line2D([], [], linestyle=markers[i], color="k") for i in range(3)]
        new_handles = []
        for i in range(len(handles)):
            if i not in [0, 6]:
                new_handles.append(handles[i])
                if i <= 6:
                    new_labels[i] = "nt=" + new_labels[i]
                new_labels_2.append(new_labels[i])
        aa.legend(new_handles, new_labels_2, fontsize=8, ncols=3)
    elif tipo is not None:
        if 'dynamic' in tipo and ax is not None:
            x_size = len(df[x_column].unique().tolist())
            ax.vlines(x=[int(0.7 * x_size)], ymin=0, ymax=100, colors='silver', ls='--', lw=1,
                      label='vline_multiple - full height')

            if 'dynamic_2' in tipo:
                if "0.1" in title:
                    a = "0.1"
                    b = "1.0"
                else:
                    a = "1.0"
                    b = "0.1"
                if "WISDM-P" in title and a == "1.0":
                    frac = 0.45
                else:
                    frac = 0.3
                print("tt: ", title)
                ax.text(int(frac * x_size), 90, r"$\alpha$=" + a, fontsize=11)
                ax.text(int(0.8 * x_size), 90, r"$\alpha$=" + b, fontsize=11)
                ax.set_title(title.split(";")[0])

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        print("chh", base_dir)
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "pdf/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "pdf/" + file_name + log + ".pdf", bbox_inches='tight', dpi=400)

def stacked_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    file_name = """{}_stacked""".format(file_name)

    if ax is None:
        # fig, ax = plt.subplots()
        figure = plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    # x = df[x_column].tolist()
    # # plt.xticks(np.arange(0, max(x) + 1, 2.0))
    #
    labels = df.columns.tolist()
    x = df[x_column].tolist()
    # args =
    # plt.stackplot(x, df[y_column].tolist(), labels=df[hue].tolist())
    # total_y = df.groupby(x_column)[y_column].sum()
    # df[y_column] = df[y_column]/total_y
    palette = sns.color_palette()

    # (
    #     so.Plot(df, x_column, y_column, color=hue)
    #     .add(so.Area(alpha=.1), so.Stack()).save(loc=base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400).save(loc=base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)
    # )
    # df = df.set_index(x_column)
    def group(df, x_column, hue):

        alphas = df[x_column].unique().tolist()
        classes = df[hue].unique().tolist()
        classes = sorted(classes, reverse=True)
        n = len(classes)
        labels = {10:[[9, 10], [7, 8], [5, 6], [3, 4], [1, 2]], 47: [[40, 47], [30, 39], [20, 29], [10, 19], [1, 9]], 43: [[39, 43], [29, 38], [19, 28], [9, 18], [1, 8]]}
        labels_str = {k1: ["""[{}, {}]""".format(str(labels[k1][i][0]), str(labels[k1][i][1])) for i in range(len(labels[k1]))] for k1 in labels}
        print("labels str")
        print(labels_str)
        classes_curves = []
        for i in range(len(labels[n])):
            curve = []
            start = labels[n][i][0]
            end = labels[n][i][1]

            for alpha in alphas:
                print("pergunta")
                query = """\u03B1=={} and (Unique_classes>={} and Unique_classes<={})""".format(alpha, start, end)
                print(query)
                print("resultado")
                print(df.query(query)['Total_of_clients_(%)'].sum().tolist())
                curve.append(df.query(query)['Total_of_clients_(%)'].sum().tolist())

            classes_curves.append(curve)

        print("curvas")
        print(classes_curves)

        return alphas, classes_curves, labels_str[n]

    x_data, curves, classes = group(df, x_column, hue)

    print("indice")
    print(df)
    # plt.stackplot(x_data, curves, labels=classes, alpha=0.8)
    classes_new = []
    for i in range(len(classes)):
        # print("metricas")
        # print(len(classes))
        # print(len(curves))
        # print(len(labels))
        # print(len(x_data))
        # plt.bar(x_data, curves[i], label=classes[i])
        classes_new += [classes[i]] * len(curves[i])
        pass

    curves = np.array(curves).flatten()
    # classes = np.array(classes).flatten().tolist() * int(len(curves)/len(classes))
    n = len(curves)
    x_data = x_data * int(n/len(x_data))
    print(len(x_data), len(curves), len(classes_new))
    df = pd.DataFrame({x_column: x_data, hue: classes_new, y_column: curves})
    sns.barplot(df, x=x_column, y=y_column, hue=hue)
    plt.legend(loc='upper right', fontsize='large', title=hue.replace("_", " "), bbox_to_anchor=(1.2, 1.1))
    plt.xlabel(x_column)
    plt.ylabel('Total of clients (%)')
    plt.title(title.replace("CIFAR10", "CIFAR-10"))
    #
    plt.show()

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if type == 2:
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
        plt.xticks(np.arange(0, max(x)+1, 20))
        plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        # so.Plot().save(loc=base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        # so.Plot().save(loc=base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)
        plt.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        plt.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def hist_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, style=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    file_name = """{}_lineplot""".format(file_name)

    if ax is None:
        # fig, ax = plt.subplots()
        figure = plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if y_lim:
        plt.ylim([y_min, y_max])

    x = df[x_column].tolist()
    # plt.xticks(np.arange(0, max(x) + 1, 2.0))

    if type is not None:
        palette = sns.color_palette()
        figure = sns.histplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, multiple="stack").set_title(title)
    else:
        figure = sns.histplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, multiple="stack").set_title(title)
    print("nof")

    # plt.xticks(np.arange(min(x), max(x) + 1, max(x)//10))
    if type == 2:
    #     plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')
        plt.xticks(np.arange(0, max(x)+1, 20))
        plt.legend(bbox_to_anchor=(1.05, 1.15), loc='right', borderaxespad=0, ncol=4, title='')
        # lines_labels = [["100"], ["10"], ["5"], ["2"], ["1"]]
        # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        # plt.legend([3, 2, 1], label='Line 1', loc='upper left', ncol=3, bbox_to_anchor=(0.2, 1))

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def heatmap_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if type is not None:
        palette = sns.color_palette()
        figure = sns.heatmap(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette, hue_order=hue_order, style=hue).set_title(title)
    else:
        figure = sns.lineplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, hue_order=hue_order, style=hue).set_title(title)

    if type == 2:
        plt.legend(bbox_to_anchor=(0.5, 1), loc='upper left', borderaxespad=0, title='Rounds since the last training (nt)')

    if type == 3:
        figure.legend([l1, l2, l3, l4],  # The line objects
                   labels=line_labels,  # The labels for each line
                   loc="center right",  # Position of legend
                   borderaxespad=0.1,  # Small spacing around legend box
                   title="Legend Title"  # Title for the legend
                   )

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

def ecdf_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, log_scale=False, ax=None, type=None, hue_order=None, y_lim=False, y_min=0, y_max=1, n=None):
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if ax is None:
        plt.figure()
    sns.set(style='whitegrid')
    log = ""
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if type is not None:
        palette = sns.color_palette()
        figure = sns.ecdfplot(x=x_column, data=df, hue=hue, ax=ax, palette=palette).set_title(title)
    else:
        figure = sns.ecdfplot(x=x_column, data=df, hue=hue, hue_order=hue_order, ax=ax).set_title(title)

    # sns.set(style='whitegrid', palette=palette)

    if ax is None:
        figure = figure.get_figure()
        Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
        Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)