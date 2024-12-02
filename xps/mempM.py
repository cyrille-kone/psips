import json
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from ext.plot_figs import get_default_fig, save_fig, __COLORS__, __MARKERS__, __NAMES__

__fname__ = "mempM"
'''@ format 
        r_tens(tt, 0ul) = (dt) m;
        r_tens(tt, 1ul) = (dt) mp;
        r_tens(tt, 2ul) = iterator_eql(ps_mask, b_ref.ps_mask);
        r_tens(tt, 3ul) = cpt_duration(s, e, std::chrono::nanoseconds);
        r_tens(tt, 4ul) = cpt_duration(ss, ee, std::chrono::nanoseconds);
        r_tens(tt, 5ul) = (dt) I_t;
'''
savefig = True
num_points = 100
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
    delta = parsed["meta"]["delta"]
    tps = np.arange(K, len(data[0][1][1]) + K)
    fig = get_default_fig()
    i=0
    for qty in [0, 1]:
        # convert to millis
        y = avg_data[i][1][:, qty]
        std = np.std(data[i][1][:, :, qty] , axis=0)
        #print(std)T+K-1
        idx = np.linspace(0,T-1, min(num_points, T)).astype(int)
        q10 = np.quantile(data[i][1][:, :, qty], 0.1, axis=0)
        q90 = np.quantile(data[i][1][:, :, qty], 0.9, axis=0)
        q50 = np.quantile(data[i][1][:, :, qty], 0.5, axis=0)
        #plt.plot(idx, q50[idx-K], label="q50")
        fig.gca().plot(tps[idx], y[idx], label=[r"$m_{t}$", r"$m_{t, \delta}$"][qty],  color=sns.color_palette("bright")[qty], linestyle=["--", "-."][qty])
        fig.gca().fill_between(tps[idx], q10[idx], q90[idx], alpha=0.2)
    fig.gca().plot(tps[idx], (1 / delta) * np.log(tps[idx] / delta), label=r"$M(t,\delta)$", linestyle=":")
    #plt.fill_between(np.arange(K, T+K), y-std, y+std,alpha=0.2)
    fig.gca().set_ylabel(r"average number of rejection samples")
    fig.gca().set_xlabel(r"iteration")
    fig.gca().legend()
    if savefig:
        save_fig(fig, f"{__fname__}.pdf")
