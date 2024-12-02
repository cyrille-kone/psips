import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ext.plot_figs import group_boxp, save_fig, __NAMES__
__fname__ = "" #"abayes_bern_d2_K40" fill
__tns_fname__ = "" #res_tns_bayes_gaussian # leave unused
load_tns = False
savefig = False
compute_tns = False
base_dir = Path("../out/appx/bayes")
if __name__=="__main__":
    # load the file
    data = []
    __fnames__ = [__fname__, __tns_fname__] if load_tns else [__fname__]
    for _fname_ in __fnames__:
        with open(base_dir / f"{_fname_}.json") as f:
            parsed = json.load(f)
            for key, val in parsed["algo"].items():
                data += [(key, np.array(val["result"]))]
                print(key, np.mean(val["result"], 0))
    fig = group_boxp(*[(__NAMES__[name], result[:, 1]) for (name, result) in data])
    if savefig:
        save_fig(fig, base_dir/f"{__fname__}.pdf")
