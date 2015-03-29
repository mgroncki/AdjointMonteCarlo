// AdjointMonteCarlo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <vector>
#include <cppad/cppad.hpp>
#include <random>
#include <string>
#include <boost/timer.hpp>
#include <iostream>
#include <ql/pricingengines/blackcalculator.hpp>

using namespace std;
using CppAD::AD;

namespace CppAD {

    inline double CondExpEq(double x, double y, 
       double a, double b) { return x == y ? a : b; }

    template<class Base> 
    CppAD::AD<Base> max(CppAD::AD<Base> x,CppAD::AD<Base> y) {
        return CppAD::CondExpGt(x,y,x,y);
    }

    template<class Base>
    Base max(Base x, Base y) {
        return CppAD::CondExpGt(x,y,x,y);
    }
}



template<typename T = double>
T PayOff(const T & S, double K){
    return CppAD::max<T>(S-K, 0.0);
}

template<typename T = double>
T BlackProcess(const T & S0,const T & sigma,const T & r, double t,  double dZ){
    return S0*exp((r - sigma*sigma / 2.0)*t + sigma*sqrt(t)*dZ);
}

template<typename T = double>
T monteCarloSim(const T & S0, const T & sigma, const T & r, double t, double K, double N = 100000) {
    std::random_device rd;
    std::mt19937 rengine(rd());
    std::normal_distribution<> normal_dist(0, 1);
    T res = 0;
    for (int i = 0; i < (int)N; i++)
        res += PayOff(BlackProcess(S0, sigma, r, t, normal_dist(rengine)), K);
    return exp(-r*t) * res / N;
}

vector<double> classicalBumpMethod(double S, double sigma, double r, double t, double K, double N = 10000000){
    vector<double> res(5, 0.);
    // Calculate NPV
    res[0] = monteCarloSim(S, sigma, r, t, K,N);
    // Calc delta
    double h = 1;
    res[1] = (monteCarloSim(S+h, sigma, r, t, K, N)-monteCarloSim(S-h, sigma, r, t, K,N)) / (2*h);
    // Calc vega
    h = 0.001;
    res[2] = (monteCarloSim(S, sigma+h, r, t, K, N)-monteCarloSim(S, sigma-h, r, t, K,N)) / (2*h);
    // Calc rho
    res[3] = (monteCarloSim(S, sigma, r+h, t, K, N )-monteCarloSim(S, sigma, r-h, t, K,N)) / (2*h);
    // Calc gamma
    h = 1;
    res[4] = (monteCarloSim(S + 2. * h, sigma, r, t, K, N) - 
            2.0 * res[0] + monteCarloSim(S - 2. * h, sigma, r, t, K, N)) 
        / 4.0*h*h ;
    return res;
}

void outputResults(const vector<double> & results){
    cout << "Delta " << results[1] << endl;
    cout << "Gamma " << results[4] << endl;
    cout << "Vega " << results[2] << endl;
    cout << "Rho " << results[3] << endl;
    cout << "NPV" << results[0] << endl;
}

vector<double> pathwiseGreeksWithAutomaticDiff(double S, double sigma, double r, double t, double K, double N){
    vector<double> results(5,0.);
    size_t m = 3;
    vector<AD<double> > x(m);
    x[0] = S; // Spot
    x[1] = sigma; // implied Vola
    x[2] = r; // risk free rate
    // Start record
    cout << "Start taping the Monte Carlo Simulation" << endl;
    boost::timer tim;
    CppAD::Independent(x);
    
    vector<AD<double> > y(1);
    y[0] = monteCarloSim(x[0], x[1], x[2], t, K, N);// Calc vega

    CppAD::ADFun<double> f(x, y);
    cout << "Time elapsed :" << tim.elapsed() << endl;
    // End record
    // Calculate the first order greeks
    vector<double> x0 = {S, sigma, r };
    double npv = f.Forward(0, x0)[0];
    vector<double> x1 = { 0., 0., 0. };
    x1[0] = 1.0;
    double delta = f.Forward(1, x1)[0];
    x1[0] = 0;
    x1[1] = 1.;
    double vega = f.Forward(1, x1)[0];
    x1[1] = 0;
    x1[2] = 1.;
    double rho = f.Forward(1, x1)[0];
    
    // Calculation 2nd derivitives
    // with bump and revaluation
    double h = 1;
    x0[0] += h;
    f.Forward(0,x0);
    x1[0] = 1;
    x1[2] = 0;
    double gamma = (f.Forward(1, x1)[0] - delta) / h;
    x0[0] -= h;
    results[0] = npv;
    results[1] = delta;
    results[2] = vega;
    results[3] = rho;
    results[4] = gamma;
    return results;
}


int main(){
    cout << "Test automatic differentation" << endl
        << "Record whole MC Simulation " << endl;
    boost::timer t;
    double S = 100.;
    double sigma = 0.1;
    double r = 0.03;
    double T = 1.;
    double K = 103.;

    QuantLib::BlackCalculator analytic(QuantLib::Option::Call, 
            K,
            S*exp(r*T),
            sigma * sqrt(T),
            exp(-r*T));

    cout << "Analytic BlackScholes Formula " << endl
        << "Price : " << analytic.value() << endl
        << "Delta : " << analytic.delta(S) << endl
        << "Gamma : " << analytic.gamma(S) << endl
        << "Vega  : " << analytic.vega(T) << endl
        << "Rho   : " << analytic.rho(T) << endl
        << endl;

    vector<double> Ns = { 1e4 , 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 5e6, 7.5e6, 1e7, 2e7, 3e7 };
    cout << "Calculate pathwise greeks with automatic differentiation " << endl;
    for(auto N : Ns) {
        cout << "Sample size = " << N << endl;
        t.restart();
        vector<double> res = pathwiseGreeksWithAutomaticDiff(S, sigma, r, T, K, N);
        outputResults(res);
        cout << "Time elapsed " << t.elapsed() << endl << endl;
    }
    cout << "Calculate pathwise greeks with finite difference " << endl;
    for(auto N : Ns) {
        cout << "Sample size = " << N << endl;
        t.restart();
        vector<double> res = classicalBumpMethod(S, sigma, r, T, K, N);
        outputResults(res);
        cout << "Time elapsed " << t.elapsed() << endl << endl;
    }
    return 0;
}

