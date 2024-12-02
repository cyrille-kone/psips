r"""
THIS FILE IS TAKEN FROM THE PUBLIC REPOSITORY @ https://github.com/elise-crepon/sequential-pareto-learning-experiments
WITH MAJOR MODIFICATIONS
"""

'''
This programs implements the Track-and-Stop algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt

from .pareto_2d import PC2d
from .pareto_nd import PCnd
from .hedge import hedge, hedge_step

'''
Algorithm of crepon et al 2024 @ https://github.com/elise-crepon/sequential-pareto-learning-experiments
with added seeding
'''


def track_and_stop(μ, lδ, tracking='D', \
                   T=None, silent=False, speedup=False, seed=42, cov=None, is_bern=False, re_norm=False):
    creator = lambda λ: PC2d(λ) if speedup else PCnd(λ)
    K = np.size(μ, axis=0)
    np.random.seed(seed)  # added line
    cov = np.eye(np.size(μ, axis=1)) if cov is None else cov  # add line
    #assert np.all(np.diagonal(cov) == cov[0][0]), "model should be homoscedastic"
    if is_bern: assert np.all(
        np.logical_and((μ <= 1), (μ >= 0))), f"means should be in (0, 1) for bernoulli but provided is {μ}"
    norm = cov[0][0]
    # normalize to be unit variance
    Σ_hat, N_hat, t = np.zeros(np.shape(μ)), np.zeros(K), 0
    N_hedge, w_hedge, reg = np.ones(K), np.ones(K) / K, 0.
    # Estimate problem difficulty
    sampler = (lambda k: np.random.binomial(1, μ[k])) if is_bern else (lambda k: np.random.multivariate_normal(μ[k], cov) / (np.sqrt(np.diag(cov)) if re_norm else 1.))
    if T is None:
        g, _, _ = hedge(μ, w_hedge.copy(), 300)
        T = int(np.ceil(lδ / g))
    # Create first estimate for Σ_hat
    for k in range(K):
        Σ_hat[k] = sampler(k)  #line modified
        t += 1
        N_hat[k] += 1
    eps = 1e-9
    μ_hat = np.copy(Σ_hat) #+ np.random.normal(loc=0,scale=eps, size=μ.shape) # avoid bug of equality in self.front being empty
    cloud_hat = creator(μ_hat)
    #print(Σ_hat.shape,μ_hat.shape, cloud_hat.get_cost(w_hedge))
    while True:
        if t%16==0 or t==K:
            g_round, grad_round = cloud_hat.get_cost(w_hedge)
            g_round /= norm
            grad_round /= norm
        reg = np.maximum(reg, np.max(np.abs(grad_round))) + eps
        hedge_step(w_hedge, grad_round, t, reg)
        if tracking == 'C':
            N_hedge += w_hedge
        elif tracking == 'D':
            N_hedge = t * w_hedge

        k_t = np.argmin(N_hat) if np.any(N_hat <= np.sqrt(t) - K / 2) else \
            np.argmax(N_hedge - N_hat)
        Σ_hat[k_t] += sampler(k_t)  # modified line
        N_hat[k_t] += 1
        μ_hat = Σ_hat / N_hat[:, np.newaxis] #+ np.random.normal(loc=0,scale=eps, size=μ_hat.shape)
        cloud_hat = creator(μ_hat)
        if t%32==0 or t==K:
            Z_t, _ = cloud_hat.get_cost(N_hat / t)
            # normalization
            Z_t /= norm
            Z_t = t * Z_t - np.log(np.log(1 + t))
            stop = Z_t >= lδ
        if not silent and (t == 1 or stop or t % 20 == 0):
            print(f'{t}/{T}: {Z_t:.3f} {">" if stop else "≤"} {lδ:.3f}')
        if stop: break
        t += 1

    correct_answer = creator(μ).front[0]
    answer = cloud_hat.front[0]
    correct = np.shape(answer) == np.shape(correct_answer) and np.all(answer == correct_answer)
    return answer, correct, t


arms_means = np.array([[9.50479943, 6.85646198, 4.56226268],
                       [9.29302574, 6.64118217, 4.03600899],
                       [9.05368656, 6.40687999, 3.56388296],
                       [10.21251518, 7.48941208, 4.42843301],
                       [10.04680837, 7.19967835, 4.36182393],
                       [8.34379173, 5.66642669, 3.51154544],
                       [8.22174773, 5.45532112, 3.64021428],
                       [9.74560492, 7.2730926, 4.7095302],
                       [10.42726889, 7.61035762, 4.71849887],
                       [8.93761259, 6.18826412, 3.84374416],
                       [7.80669637, 5.26269019, 3.97029191],
                       [8.85008761, 6.58892648, 4.7335634],
                       [8.4411757, 6.15273269, 4.5890408],
                       [9.92900909, 7.39079852, 4.74927053],
                       [9.68315255, 7.20340552, 4.91191932],
                       [7.51479976, 5.3082677, 3.95508249],
                       [7.26542972, 4.99043259, 4.01638302],
                       [8.61558951, 6.33327963, 4.66343909],
                       [10.34531673, 7.769801, 5.00327494],
                       [8.29304914, 5.92157842, 3.86702564]])

covariance = np.diag(np.array([0.70437039, 0.82845749, 1.53743137]))
from time import time
if __name__ == '__main__':
    K, d, lδ = 20, 3, np.log(1. / 1e-2)
    μ = arms_means
    start = time()
    answer, correct, t = track_and_stop(μ, lδ, cov=covariance, re_norm=True, silent=True)
    print(time() - start, t, correct)