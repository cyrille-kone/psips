import json
import numpy as np
from pathlib import Path
import seaborn as sns
from ext.lbd import cpt_lbd_correl
import matplotlib.pyplot as plt
from ext.plot_figs import group_boxp, save_fig, __NAMES__
savefig = True
__fname__ = "mnoc"
excluded_keys = ["RR"]
if __name__ == "__main__":
    # load the file
    data = []
    delta = None
    with open(Path() / f"{__fname__}.json") as f:
        parsed = json.load(f)
        delta = parsed["meta"]["delta"]
        print(delta)
        for key, val in parsed["algo"].items():
            if key not in excluded_keys:
              data += [(key, np.array(val["result"]), val["duration"])]
    print(*[(data[i][0], data[i][1].mean(0)) for i in range(len(data))])
    fig = group_boxp(*[(__NAMES__[name], result[:, 1]) for (name, result, _) in data])
    #fig.gca().ticklabel_format(axis='y', style='scientific', scilimits=(1, 4),useMathText=True, useLocale=True)
    if savefig:
        save_fig(fig, f"{__fname__}.pdf")
    fig.show()
