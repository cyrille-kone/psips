#pragma once
#include <cmath>
#include<vector>
#include<cstddef>
#include <numeric>
#include<iostream>
#include<cmath>
#include <algorithm>
#include<xtensor.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <valarray>
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
using json = nlohmann::json;
#define iterator_eql(it1, it2) \
std::equal((it1).begin(), (it1).end(), (it2).begin(), (it2).end())

#define return_cr_sc(ret_mask, ps_mask, Nt)  \
{iterator_eql(ret_mask, ps_mask), std::accumulate(Nt.begin(), Nt.end(), 0ul)}

#define upt_means(mu, total_outcome, arm, vcount, b_ref) \
{xt::row((total_outcome), (arm)) += xt::row((b_ref).sample({(arm)}), 0ul); \
(vcount)(arm) += 1ul;                                       \
xt::row((mu), (arm)) = xt::row((total_outcome), (arm)) / (dt)(vcount)(arm);}

#define cpt_duration(start, end, unit) \
std::chrono::duration_cast<unit>((end)-(start)).count()

//#pragma GCC optimize ("O1")
#define duration_of_instruction(instr, result, duration) \
        {auto clock  = std::chrono::steady_clock::now();       \
        {result = instr;}                                    \
        duration = cpt_duration(clock, std::chrono::steady_clock::now(), std::chrono::nanoseconds);}
using dt = double;
thread_local inline std::default_random_engine gen(42ul);
//thread_local inline std::normal_distribution<dt> r_g(0.); //
#define INF (1e7)
// maximum number of arms that can be handled
#define MAX_K 100000
#define EE 2.71828182845904523536
#define get_argmin(v, idx) (idx)[std::distance((v).begin(), std::min_element((v).begin(), (v).end()))]
#define get_argmax(v, idx) (idx)[std::distance((v).begin(), std::max_element((v).begin(), (v).end()))]
#define in_set(id, v) (std::find((v).begin(), (v).end(), (id)) != (v).end())
template <typename T=dt, size_t s=1>
using vv = xt::xtensor<T, s>;

 inline auto fpsi(const vv<dt,2>& x){
            bool is_dom;
            size_t K{x.shape()[0]};
            vv<bool> is_opt = xt::ones<bool>({K});
            for(size_t i{0}; i<K; ++i){
                is_dom = false;
                for(size_t j{0}; j<K; ++j){
                    if (i == j || !is_opt[j]) continue;
                    is_dom = xt::all(xt::row(x, i)<= xt::row(x, j)) && xt::any(xt::row(x, i)< xt::row(x, j)) ; // true if mu_i is dominated by mu_j
                    if (is_dom) {
                        break;}
                }
                is_opt[i] = (!is_dom);
            }
            return is_opt;
        }
inline auto ffpsi(const vv<dt,2>& x){
    return fpsi(x);
    /*
    auto K{x.shape()[0]};
    //auto ps =  new size_t[K];
    vv<bool> ps_mask = xt::zeros<bool>({K});
    // argsort according to first axis
    vv<size_t> idx {xt::arange(K)};
    // sort in descending order
    std::sort(idx.begin(), idx.end(),[&x](auto a, auto b){return x(a, 0)> x(b, 0);});
    //ps[count] = idx[count];
    ps_mask[idx[0]] = true;
    dt curr_max = x(idx(0), 1);
    for (size_t i =1; i<K; ++i){
        ps_mask(idx(i)) = curr_max < x(idx(i), 1);
        curr_max = std::max(curr_max, x(idx(i), 1));
    }
    return ps_mask;*/
    /*++count;
    bool sub;
    // smart checking will reduce the second loop
    for (size_t i =1; i<K; ++i){
        sub = false;
        for (size_t j =0; j<count; ++j){
            if (x(idx[i], 1) < x(ps[j], 1)){
                // this arm is sub-optimal
                sub = true;
                break ;
            }
        }
        if (!sub) {
            ps[count] = idx[i];
            ps_mask[idx[i]] = true;
            ++count;}
    }
    return ps_mask;*/

}

inline bool is_pareto_dominated(const vv<double>& xi, const vv<double>& xj, const double& eps=0.){
    // return true if xi is dominated by xj
    return xt::all(xi +eps <= xj) && xt::any(xi+eps<xj);
};

/*
 * utility function for the implementation of APE
 */
namespace utils::ape {
    inline dt cpt_index_ps(const size_t &a, const vv<dt,2ul> &means, const vv<dt> &fs) {
        return xt::amin(xt::amax(xt::view(xt::row(means, a), xt::newaxis(), xt::all()) - means, {1}) - fs +
                        INF * xt::row(xt::eye<dt>(means.shape()[0]), a))(0u) - fs(a);
    }

    inline dt cpt_index_non_ps(const size_t &a, const vv<dt,2ul> &means, const vv<dt> &fs) {
        return xt::amax(xt::amin(means - xt::view(xt::row(means, a), xt::newaxis(), xt::all()), {1}) - fs -
                        INF * xt::row(xt::eye<dt>(means.shape()[0]), a))(0ul) - fs(a);
    }

    inline size_t get_bt(const vv<dt, 2> &means, const vv<bool> &ps_mask, const vv<dt> &fs) {
        auto x = xt::empty<dt>({means.shape()[0]});
        for (size_t i{0}; i < means.shape()[0]; ++i) {
            x(i) = ps_mask(i) ? cpt_index_ps(i, means, fs) : cpt_index_non_ps(i, means, fs);
        }
        return xt::argmin(x)(0);
    }

    inline size_t get_ct(const vv<dt, 2> &means, const size_t bt, const vv<dt> &fs) {
        // compute the  Mij for a list of
        vv<dt> x = xt::amax(xt::view(xt::row(means, bt), xt::newaxis(), xt::all()) - means, {1ul}) - fs +
                   INF * xt::row(xt::eye<dt>(means.shape()[0]), bt);
        return xt::argmin(x)(0);
    }

    [[maybe_unused]] inline dt cpt_z1_z2(const vv<dt,2ul> &means, const vv<bool> &ps_mask, const vv<dt> &fs) {
        dt xmin{1.0};
        for (size_t i{0}; i < means.shape()[0]; ++i) {
            xmin = std::min(xmin, ps_mask(i) ? cpt_index_ps(i, means, fs) : cpt_index_non_ps(i, means, fs));
        }
        return xmin;
    }

    inline dt beta(const size_t &t, const dt &delta) {
        static dt log_d = 2.* log((dt) (1.) / delta) + 3.*std::max(log(log((dt) (1.) / delta)), 0.);
        return sqrt((log_d + log(1.+ log((dt)t)))/ (dt)t);
    }
}

/*
 * utility function for the implementation of peps for PSI
 */

namespace utils::peps{
    // generate a new parameter
    inline auto gen_i(const vv<dt,2>& m, const vv<dt,2>& c_cov, const vv<size_t>& cs, dt eta=1.)  {
        auto mtx = xt::empty_like(m);
        auto ptr = mtx.data();
        auto d = m.shape()[1];
        for (size_t i{0}; i<m.size(); ++i){
            ptr[i] = std::normal_distribution<dt>(0.)(gen) / (std::sqrt(cs(i/d))*eta);}
        mtx = xt::linalg::dot(mtx, c_cov) + m;
        return mtx;
    }

    // check if the ps of the parameter lda and ps_mask differs
    inline auto in_alt(const vv<dt,2>& lda, const vv<bool>& ps_mask){
        bool dom_f;
        size_t s {ps_mask.size()}, j;
// # pragma omp parallel for
        for(size_t i{0}; i<s; ++i){
            /* if arms in ps are not dominated by each other for the param $\lambda$
             * check if any arm not in ps in dominated by an arm of ps
             */
            dom_f = false;
            for (j=0; j<s; ++j){
                if (i==j) continue;
                if (ps_mask[i] && ps_mask[j] && is_pareto_dominated(xt::row(lda, i), xt::row(lda, j))) return true;
                if (!ps_mask[i] && ps_mask[j]){
                    dom_f = dom_f || is_pareto_dominated(xt::row(lda, i), xt::row(lda, j)); // || and &&  are lazy-evaluated
                }
            }
            if (!ps_mask[i] && !dom_f) return true; // arm $i$ is not in the candidate ps and no arm in it dominates $i$
        }
        return false;
    }

    inline auto struct_in_alt(const vv<dt,2>& theta, const vv<dt,2>& A, const vv<bool>& ps_mask){
        bool dom_f;
        size_t s {ps_mask.size()}, j;
// # pragma omp parallel for
        for(size_t i{0}; i<s; ++i){
            /* if arms in ps are not dominated by each other for the param $\lambda$
             * check if any arm not in ps in dominated by an arm of ps
             */
            dom_f = false;
            for (j=0; j<s; ++j){
                if (i==j) continue;
                if (ps_mask[i] && ps_mask[j] && is_pareto_dominated(xt::linalg::dot(theta, xt::row(A, i)), xt::linalg::dot(theta, xt::row(A, j)))) return true;
                if (!ps_mask[i] && ps_mask[j]){
                    dom_f = dom_f || is_pareto_dominated(xt::linalg::dot(theta, xt::row(A, i)), xt::linalg::dot(theta, xt::row(A, j))); // || and &&  are lazy-evaluated
                }
            }
            if (!ps_mask[i] && !dom_f) return true; // arm $i$ is not in the candidate ps and no arm in it dominates $i$
        }
        return false;
    }
    inline double M(const size_t &t, const dt & delta ){
    return (1./log(1./(1.-delta)))*log((dt)t/delta);
        //static dt d_inv = (1./delta);
       //return d_inv*log((dt)t*d_inv) ; // stylized version
    }
    inline double c(const size_t & t, const dt& delta){
        return 1. + log(std::max(log((dt)(t)), 1.)) / log(1./delta);
    }
}
// utils for psi
namespace utils::psi{
    template <typename T, typename V>
    inline auto M( T& x, V&y ){
        return xt::amax(x-y, {-1});
    }
    template <typename T, typename V>
    inline auto m(T& x, V&y){
        return xt::amin(y - x, {-1});
    }
    inline vv<dt> delta_star(const vv<dt,2>& means){
        return xt::amax( xt::amin((means - xt::view(means, xt::all(), xt::newaxis(),  xt::all())),-1), -1);
    }
    // compute the the min of $\delta_i^+, \delta_i^-$ as appears in Auer et al 2016
    [[maybe_unused]] inline double subopt_gap_dpm(size_t i, const vv<dt,2>& means, const vv<dt,1>& vec_delta_star) {
        double xres{INF};
        for (size_t j{0}; j<means.shape()[0];++j){
            xres = std::min(xres, INF*(i==j) + std::min(xt::amax(xt::row(means,i)-xt::row(means,j))(0ul), std::max(xt::amax(xt::row(means,j)- xt::row(means, i))(0ul), 0.) + std::max(vec_delta_star[j], 0.)+INF*(i==j)));
        }
        return xres;
    }
    inline vv<dt> compute_gaps(const vv<dt,2>& arms_means){
        size_t K{arms_means.shape()[0]};
        vv<dt> gaps = xt::empty<dt>({K});
        auto d_star = delta_star(arms_means);
        for(size_t i=0; i<K; ++i){
            gaps(i) = std::max(d_star(i), subopt_gap_dpm(i,arms_means, d_star));
        }
        return gaps;
    }
    inline dt compute_cplxty(const vv<dt,2>& arms_means){
                return xt::sum(1./ xt::pow(compute_gaps(arms_means), 2.))(0ul);}
}