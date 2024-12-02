

#include "ada_hedge.hpp"

// implementing AdaHedge
ada_hedge::ada_hedge(const size_t &K): K(K), logK(log((dt)K)) {
    init_learner();
}
void ada_hedge::init_learner() {
    L_t = xt::zeros<dt>({K});
    w_t = xt::ones_like(L_t)/(dt) K;
    Delta_t = 0.01; //1./INF;
}

void ada_hedge::update_learner(vv<> &loss) {
    eta_t = logK/Delta_t;
    // compute current weights w_t
# define CPT_CURRENT {mn = *std::min_element(L_t.begin(), L_t.end()); \
    (eta_t>=INF)? w_t = xt::equal(L_t, mn) : w_t = xt::exp(-eta_t*(L_t - mn));\
    s = std::accumulate(w_t.begin(), w_t.end(),0.); \
    w_t = w_t/s;}
    CPT_CURRENT
    // TODO check
    M_t_prev = mn - (1./eta_t)*(log(s) - logK);
    //M_t_prev = *std::min_element(loss.begin(), loss.end()) - log(s/(double)K)/eta_t;
    h_t = xt::linalg::dot(w_t, loss)(0);
    L_t += loss;
    mn = *std::min_element(L_t.begin(), L_t.end());
    M_t = mn - log(
            xt::sum((eta_t>=INF)?vv<>{xt::equal(L_t, mn)}: vv<>{xt::exp(-eta_t*(L_t - mn))})(0)/(double)K)/eta_t;
    Delta_t += std::max(0., h_t - (M_t- M_t_prev));
    CPT_CURRENT

}
vv<>& ada_hedge::operator()(void){
    // compute new weight vector
    //CPT_CURRENT
    return w_t;
}
/*
vv<double> ada_hedge_func(vv<double, 2>& losses){
    // number of experts and their losses
    auto T = losses.shape()[0];
    auto K = losses.shape()[1];
    double eta, delta, M, Mprev, Delta = 1./INF;
    vv<double> tmp{xt::empty<double>({K})};
    vv<double> L{xt::zeros<double>({K})}, w{xt::empty<double>({K})}, h{xt::empty<double>({T})};
    for(auto t=0; t<T; ++t){
        eta = log((double)K)/Delta;
        Mprev = ada_mix_loss(eta, L, w);
        h(t) = xt::linalg::dot(w, xt::row(losses, t))[{0}];
        L += xt::row(losses, t);
        M = ada_mix_loss(eta, L, tmp);
        delta = std::max(0., h(t) - (M- Mprev));
        // (max clips numeric Jensen violation)
        Delta += delta;
    }
    return w;
}

*/


vv<>& lma::operator()(void){
    // compute new weight vector
    //CPT_CURRENT
    return w_t;
}

lma::lma(const size_t &K): K(K) {
    init_learner();
}
void lma::update_learner(vv<> &loss) {
    ++t;
    eta_t = sqrt(log((dt)K)/(dt)(t));///float(self.S)
    w_t *= xt::exp(-eta_t*(loss-xt::amin(loss)));
    w_t /= xt::sum(w_t);
    gamma = 1./(dt)(4.*sqrt((dt)t));
    //w_t = (1.-gamma)*w_t+gamma/(dt)(K);
}

void lma::init_learner() {
    L_t = xt::zeros<dt>({K});
    w_t = xt::ones_like(L_t)/(dt) K;
    t = 0;
}