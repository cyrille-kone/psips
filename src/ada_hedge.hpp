

#pragma once
#include "utils.hpp"


/*inline auto ada_mix_loss(double& eta, vv<double>& L, vv<double>& w){
    auto mn = *(std::min_element(L.begin(), L.end()));
    double s, M;
    (eta>=INF)?w = xt::equal(L, mn) : w = xt::exp(-eta*(L - mn));
    s = std::accumulate(w.begin(), w.end(),0.);
    w = w/s;
    M = mn - log(s/(double)L.size())/eta;
    //std::cout<<"loss: "<<w<<"\n";
    return M;
}
 */
//vv<double> ada_hedge_func(vv<double,2>&);

struct ada_hedge {
    const size_t K;
    const dt logK;
    double h_t;
    double eta_t, mn, s,  M_t, M_t_prev, Delta_t;
    vv<> L_t, w_t;
    explicit ada_hedge(const size_t&);
    ada_hedge() = delete;
    void update_learner(vv<>&);
    void init_learner();
    vv<>& operator()(void);
};

struct  lma{
    const size_t K;
    double h_t;
    size_t t{0};
    dt gamma;
    double eta_t, mn, s,  M_t, M_t_prev, Delta_t = 1/INF;
    vv<> L_t, w_t;
    explicit lma(const size_t&);
    lma() = delete;
    void update_learner(vv<>&);
    void init_learner();
    vv<>& operator()(void);

};