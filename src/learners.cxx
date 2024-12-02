#include "learners.hpp"
void ada_hedge::init() {
    Lcum_t = xt::zeros<dt>({K});
    w = xt::ones_like(Lcum_t)/(dt) K;
    Delta_t = 1./INF;
}

void ada_hedge::updt(const vv<dt,1> & L) {
    eta = logK* (1./Delta_t);
# define CPT_CURRENT_W {mn = *std::min_element(Lcum_t.begin(), Lcum_t.end()); \
    (eta>=INF)? w = xt::equal(Lcum_t, mn) : w = xt::exp(-eta*(Lcum_t - mn));\
    s = std::accumulate(w.begin(), w.end(),0.); \
    w = w /s;}
    CPT_CURRENT_W
    Mprev = mn - (1./eta)*(log(s) - logK);
    h = xt::linalg::vdot(w, L);
    Lcum_t += L;
    mn = *std::min_element(Lcum_t.begin(), Lcum_t.end());
    // TODO check formula below
    M = mn - (log(
            (eta>=INF)?xt::sum(xt::equal(Lcum_t, mn))(0ul): xt::sum(xt::exp(-eta*(Lcum_t - mn)))(0ul)) - logK)*(1./eta);
    Delta_t += std::max(0., h - (M- Mprev));
    CPT_CURRENT_W
}

void lmd::init() {
    w_t = xt::ones<dt>({K})/(dt) K;
    t = 0;
}
void lmd::updt(const vv<> &L) {
    ++t;
    eta = sqrt(logK/(dt)(t));///float(self.S)
    w_t *= xt::exp(-eta*(L-xt::amin(L)));
    w_t /= xt::sum(w_t);
    gamma = 1./(dt)(4.*sqrt((dt)t));
    w_t = (1.-gamma)*w_t+gamma/(dt)(K);
};

void ftl::init() {
    std::cout<<"init from ftl \n";
    w_t = xt::ones<dt>({K})/(dt) K;
    Lcum_t = xt::zeros<dt>({K});
}
void ftl::updt(const vv<> &L) {
    Lcum_t += L;
    w = xt::equal(Lcum_t, xt::amin(Lcum_t));
    w /= xt::sum(w);
};