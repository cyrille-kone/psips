#include <vector>
#include <random>
#include "utils.hpp"
#include "bandits.hpp"
#include "xtensor.hpp"
#include <xtensor/xindex_view.hpp>
#include <xtensor-blas/xlinalg.hpp>
void bandit::reset_env(const size_t& seed) const{
    gen.seed(seed);
    // this->seed = seed;
}
bandit::bandit(const vv<dt,2>& arms_means_, const vv<dt,2>& cov_): arms_means(arms_means_), K(arms_means_.shape()[0]),
d(arms_means_.shape()[1]), action_space(xt::arange(arms_means_.shape()[0])), ps_mask(fpsi(arms_means_)),
ps(xt::filter(xt::arange(arms_means_.shape()[0]), fpsi(arms_means_))), cov(cov_), chol_cov(xt::linalg::cholesky(cov_)){
    // rng initialization with default seed
    //gen = std::mt19937_64 (42ul); // initialize thread_local variable;
}

vv<dt, 2> bandit::sample(const vv<size_t,1> &arms) const {
    return sample(std::move(arms)); // forcing the call to the auto sample (T&&)
}


bernoulli::bernoulli(const vv<dt,2>& arms_means_, const vv<dt,2>& cov_): bandit(arms_means_, cov_) {
};

gaussian::gaussian(const vv<dt,2>& arms_means, const vv<dt,2>& cov): bandit(arms_means, cov) {
}

vv<dt,2> gaussian::sample(const vv<size_t,1>&& arms) const {
    auto r_tens = xt::empty<dt>({arms.size(), d});
    for(size_t i =0; i<arms.size(); ++i){
       xt::row(r_tens, i) = xt::ravel(xt::linalg::dot(chol_cov, xt::random::randn<dt>({d}, 0., 1., gen))) + xt::row(arms_means, arms(i));
    }
    return r_tens;
}

vv<dt,2> bernoulli::sample(const vv<size_t,1ul>&& arms) const  {
    auto r_tens = xt::empty<dt>({arms.size(), d});
    for (size_t i=0; i<arms.size(); ++i){
        for(size_t cx=0ul; cx<d; ++cx)
            r_tens(arms(i), cx) = std::bernoulli_distribution(arms_means(arms(i), cx))(gen); //
    }
    return r_tens;
}