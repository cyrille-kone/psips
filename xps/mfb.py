import json
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from ext.plot_figs import get_default_fig, save_fig, __COLORS__, __MARKERS__, __NAMES__

__fname__ = "mcovboost_fb"
'''@ format 
        r_tens(tt, 0ul) = (dt) m;
        r_tens(tt, 1ul) = (dt) mp;
        r_tens(tt, 2ul) = iterator_eql(ps_mask, b_ref.ps_mask);
        r_tens(tt, 3ul) = cpt_duration(s, e, std::chrono::nanoseconds);
        r_tens(tt, 4ul) = cpt_duration(ss, ee, std::chrono::nanoseconds);
        r_tens(tt, 5ul) = (dt) I_t;
'''
savefig = True
num_points = 25
base_dir = Path() #Path("../out/main/cov_boost")
if __name__ == "__main__":
    # load the file
    data = []
    with open(base_dir / f"{__fname__}.json") as f:
        parsed = json.load(f)
        for key, val in parsed["algo"].items():
            data += [(key, np.array(val["result"]))]
    niter = len(data[0])  # number of iteration
    T = data[0][1].shape[1]  # budget
    print(T)
    avg_data = [(name, np.mean(_data_, 0)) for (name, _data_) in data]
    K = parsed["meta"]["K"]
    qty = 2
    fig = get_default_fig()
    for i in [0, 1]:
        y = avg_data[i][1][:, qty]
        std = np.std(data[i][1][:, :, qty], axis=0)
        tps = np.arange(K, T+K)
        idx = np.linspace(0, T-1, min(num_points, T)).astype(int)
        fig.gca().plot(tps[idx], 1-y[idx], label=__NAMES__[avg_data[i][0]],  color=__COLORS__[__NAMES__[data[i][0]]], marker=__MARKERS__[data[i][0]], linestyle="--")
    fig.gca().set_ylabel(r"average probability of error")
    fig.gca().set_xlabel(r"iteration")
    fig.gca().legend()
    plt.show()
    if savefig:
        save_fig(fig, f"{__fname__}.pdf")
