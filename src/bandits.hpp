#pragma once
#include <random>
#include <vector>
#include <cstddef>
#include<xtensor.hpp>
#include "utils.hpp"
struct bandit{
    const size_t K;
    const size_t d;
    // double H;
    // size_t seed{42};
    // double sigma;
    //std::mt19937 gen;
    //size_t optimal_arm; //  only defined for 1d bandit
    //std::vector<double> suboptimal_gaps;
    const vv<size_t> action_space;
    const vv<size_t> ps;
    const vv<bool> ps_mask;
    const vv<dt,2> arms_means;
    const vv<dt,2> cov;
    const vv<dt,2> chol_cov; // cholesky of the covariance matrix
    bandit() = delete;
    bandit(const vv<dt,2>&, const vv<dt,2>& cov);
    // bandit(const vv<dt,2>&);
    [[nodiscard]] virtual vv<dt,2> sample(const vv<size_t>&) const;
    [[nodiscard]] virtual vv<dt,2> sample(const vv<size_t> &&) const =0;
    virtual void reset_env(const size_t&)const;
};

struct gaussian: bandit {
    gaussian() = delete;
    gaussian(const vv<dt,2>&, const vv<dt,2>&);
    [[nodiscard]] vv<dt,2> sample(const vv<size_t> &&)const override;
};

struct bernoulli: bandit {
    bernoulli() = delete;
    bernoulli(const vv<double, 2>&, const vv<dt,2>&);
    //[[nodiscard]] vv<double,2> sample(const vv<size_t> &);
    [[nodiscard]] vv<double,2> sample(const vv<size_t> &&)const override;
};