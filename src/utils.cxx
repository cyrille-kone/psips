#include<vector>
#include <algorithm>
#include "utils.hpp"
#include <execution>
#include <numeric>
#include <xtensor/xmanipulation.hpp>



vv<bool> pareto_optimal_arms_mask(const vv<dt,2>&means){
    // return fpsi(means);
    bool is_dom;
    size_t K{means.shape()[0]};
    auto ret_value = xt::empty<bool>({K});
    for(size_t i{0}; i<K; ++i){
        is_dom = false;
        for(size_t j{0}; j<K; ++j){
            if (i == j) continue;
            is_dom = is_pareto_dominated(xt::row(means, i), xt::row(means, j), 0.); // true if mu_i is dominated by mu_j
            if (is_dom) {
                break;}
        }
        ret_value[i] = (!is_dom);
    }


    return ret_value;

}






