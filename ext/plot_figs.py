import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# color mapping
__NAMES__ = {["APE", "PSITS", "RR", "ORACLE", "TNS", "AUER"][i]: ["APE", "PSIPS", "RR", "ORACLE", "TnS", "RR+E"][i] for i in range(6)}
__COLORS__ = {__NAMES__[["APE", "PSITS", "RR", "ORACLE", "TNS", "AUER"][i]]: sns.color_palette("deep")[i] for i in range(6)}
__MARKERS__ = {["APE", "PSITS", "RR", "ORACLE", "TNS", "GEN"][i]: ["s", "v", "X", "D", "H", ">"][i] for i in range(6)}
# configure plot and return and instance of Figure
def get_default_fig():
    plt.clf()
    plt.style.use('ggplot')
    sns.set(style="ticks")
    sns.set_context("paper")
    plt.rcParams.update({"pdf.fonttype": 42,
                         "pgf.preamble": "\n".join([
                             r"\usepackage{amsfonts}",
                             r"\usepackage{bm}",
                         ])})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    # ytick.major.size:    3.5     # major tick size in points
    # ytick.minor.size:    2       # minor tick size in points
    # ytick.major.width:   0.8     # major tick width in points
    # ytick.minor.width:   0.6     # minor tick width in points

    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams.update({'xtick.minor.bottom': False})
    plt.rcParams.update({'ytick.minor.left': False})
    plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{bm}'})
    # sns.color_palette("viridis")
    fig = plt.figure(figsize=(5, 4), layout='constrained', edgecolor='black')
    ax = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.spines[axis].set_color("black")
    # plt.tight_layout()
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    return fig


def group_boxp(*tupls):
    r'''
    each tuple is expected in the format (name, data)
    :param tupls:
    :return: plt figure
    '''
    # parse to retrieve data
    data = np.array([])
    names = []
    for (_name_, _data_) in tupls:
        print(np.shape(_data_))
        data = np.concatenate((data, _data_))
        names += [_name_, ] * np.shape(_data_)[0]
    df = pd.DataFrame({"sc": data, "names": names})
    # default config can be loaded from another function
    fig = get_default_fig()
    sns.boxplot(data=df, x="names", y="sc", orient="v", showmeans=True,
                showfliers=False, notch=True,
                width=.5, linewidth=2.,
                medianprops={"color": "0.2", "linewidth": 2.},
                palette=__COLORS__,
                meanprops={'markeredgecolor': "black",
                           'markerfacecolor': 'black', 'markersize': 5},
                # whis=[10, 90]
                ax=fig.gca()).set(
        xlabel=' ',
        ylabel=r'')
    return fig

# save plot using common parameters
def save_fig(fig, fname):
    current_values = fig.gca().get_yticks()
    # processing before plotting
    fig.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    fig.savefig(fname, transparent=True, dpi=2000,
                edgecolor='black', backend="pgf", format="pdf")
    return
