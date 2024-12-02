#pragma once
#include "utils.hpp"
struct learner{
    explicit learner(size_t K):K(K), w(xt::ones<dt>({K}) / (dt)K){};
    learner()= default;
    virtual void init(void) { std::cout<<"init from base class \n"; };
    virtual void updt(const vv<dt,1>&L) {};
    virtual vv<dt,1>& getw(void){return w;};
    learner(const learner& other):K(other.K), w(xt::ones<dt>({other.K}) / (dt)other.K){
        std::cout<<"copy ctor is called from LEARNER"<<"\n";
    }
    virtual learner* cpy(){return new learner(K);};
protected:
    vv<dt,1> w;
    size_t K ;
};

struct ada_hedge: learner {
    explicit ada_hedge(const size_t& K): learner(K), logK(log((dt)K)){};
    ada_hedge() = default;
    void updt (const vv<dt,1>&) override;
    void init() override;
    ada_hedge* cpy(){
        return new ada_hedge(this->K);
    }
private:
    dt logK, h, mn, s, M, Mprev, eta, Delta_t;
    vv<dt,1> Lcum_t;
};

// Lazy Mirror Descent
struct lmd:learner{
    explicit lmd(const size_t& K): learner(K), w_t(w), logK(log((dt)K)){};
    lmd() = delete;
    void init() override;
    void updt(const vv<dt,1>& L) override;
    lmd* cpy(){
        return new lmd(this->K);
    }

private:
    size_t t{0}; // timestep counter
    dt gamma, eta, logK;
    vv<dt,1>& w_t; // w_t is an alias for w
};

// FTL
struct ftl: learner{
explicit ftl(const size_t& K): learner(K), w_t(w){};
    ftl() = delete;
void init() override;
void updt(const vv<dt,1>& L) override;
    ftl(const ftl& other):learner(other.K), w_t(other.w_t){
    std::cout<<"copy ctor is from FTL called"<<"\n";
};
    ftl* cpy(){
        return new ftl(this->K);
    }

private:
vv<dt,1>& w_t; // w_t is an alias for w
vv<dt,1> Lcum_t;
};


