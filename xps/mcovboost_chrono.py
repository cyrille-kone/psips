import json
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from ext.plot_figs import get_default_fig, save_fig, __COLORS__, __MARKERS__, __NAMES__

__fname__ = "mcovboost_chrono"
'''@ format 
        r_tens(tt, 0ul) = (dt) m;
        r_tens(tt, 1ul) = iterator_eql(ps_mask, b_ref.ps_mask);
        r_tens(tt, 2ul) = cpt_duration(s, e, std::chrono::nanoseconds);
        r_tens(tt, 3ul) = cpt_duration(ss, ee, std::chrono::nanoseconds);
        r_tens(tt, 4ul) = (dt) I_t;
'''
savefig = True
num_points = 25
if __name__ == "__main__":
    # load the file
    data = []
    with open(Path() / f"{__fname__}.json") as f:
        parsed = json.load(f)
        for key, val in parsed["algo"].items():
            data += [(key, np.array(val["result"]))]
    niter = len(data[0])  # number of iteration
    T = data[0][1].shape[1]  # budget
    print(T)
    avg_data = [(name, np.mean(_data_, 0)) for (name, _data_) in data]
    K = parsed["meta"]["K"]
    qty = 3
    fig = get_default_fig()
    for i in [0, 1]:
        # convert to millis
        cf = 1_000_000
        y = avg_data[i][1][:, qty] / cf
        std = np.std(data[i][1][:, :, qty] / cf, axis=0)
        #print(std)
        idx = np.linspace(K, T+K-1, min(num_points, T)).astype(int)
        q10 = np.quantile(np.cumsum(data[i][1][:, :, qty], -1) / cf, 0.1, axis=0)
        q90 = np.quantile(np.cumsum(data[i][1][:, :, qty], -1) / cf, 0.9, axis=0)
        fig.gca().plot(idx, np.cumsum(y)[idx-K], label=__NAMES__[avg_data[i][0]],  color=__COLORS__[__NAMES__[data[i][0]]], marker=__MARKERS__[data[i][0]], linestyle="--")
        fig.gca().fill_between(idx, q10[idx-K], q90[idx-K], alpha=0.2)
        #plt.fill_between(np.arange(K, T+K), y-std, y+std,alpha=0.2)
    fig.gca().set_ylabel(r"average cumulated time [ms]")
    fig.gca().set_xlabel(r"iteration")
    fig.gca().legend()
    if savefig:
        save_fig(fig, f"{__fname__}cumulated.pdf")
