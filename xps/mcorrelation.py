import json
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from ext.plot_figs import get_default_fig, save_fig, __COLORS__, __MARKERS__, __NAMES__
#from ext.algos import cpt_tau_tns_batch_cov, cpt_T_star_batch_cov

__fname__ = "" # "mcorrelation34"
'''@ format 
        data(num_vals_of_correl_coeff, nruns, 2) 
'''
savefig = True
compute_tns = False
load_tns = True
__tns_fname__ = "res_tns"
save_tns = False
plot_th = True

if __name__ == "__main__":
    # load the file
    data = []
    correl_coeffs = []
    __fnames__ = [__fname__, __tns_fname__] if load_tns else [__fname__]
    for _fname_ in __fnames__:
        with open(Path() / f"{_fname_}.json") as f:
           parsed = json.load(f)
           correl_coeffs = np.array(parsed["meta"]["correl_coeff"])
           for key, val in parsed["algo"].items():
               data += [(key, np.array(val["result"]))]
    nruns = data[0][1].shape[-2]
    arms_means = np.array(parsed["meta"]["arms_means"])
    covs = np.array([[[1, rho], [rho, 1]] for rho in correl_coeffs])
    fig = get_default_fig()
    """
    if compute_tns and not load_tns:
        res_tns = cpt_tau_tns_batch_cov(arms_means, covs,
                                        seeds=np.arange(10),  # temporary
                                        delta=parsed["meta"]["delta"],
                                        verbose=1)

        if plot_th:
            res_ind, res_correl = cpt_T_star_batch_cov(arms_means, covs, niter=100, nniter=100, verbose=0)
        data += [("TNS", res_tns)]"""
    for (_name_, _data_) in data:
        y = np.mean(_data_, -2)
        print(_data_.shape)
        print(min(y[:, 1]), max(y[:, 1]), y[-1, 1])
        q10 = np.quantile(_data_, 0.10, axis=-2)
        q90 = np.quantile(_data_, 0.90, axis=-2)
        std = np.std(_data_, -2)
        fig.gca().plot(correl_coeffs[:-1], y[:-1, 1], label=__NAMES__[_name_], color=__COLORS__[__NAMES__[_name_]], marker=__MARKERS__[_name_])
        #fig.gca().fill_between(correl_coeffs[:-1], q10[:-1, 1], q90[:-1, 1], alpha=0.2)
        fig.gca().fill_between(correl_coeffs[:-1], (y - std)[:-1, 1], (y + std)[:-1, 1], alpha=0.2)
    fig.gca().legend()
    fig.gca().set_ylabel(r"average sample complexity")
    fig.gca().set_xlabel(r"correlation coefficient $\rho$")
    plt.show()
    if savefig:
        save_fig(fig, f"{__fname__}.pdf")
