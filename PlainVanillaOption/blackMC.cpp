// AdjointMonteCarlo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <vector>
#include <cppad/cppad.hpp>
#include <random>
#include <string>

using namespace std;
using CppAD::AD;

template<typename T = double>
T PayOff(const T & S, double K){
    return ((S-K) + abs(S-K)) / 2.0;
}

template<typename T = double>
T BlackProcess(const T & S0,const T & sigma,const T & r, double t,  double dZ){
    return S0*exp((r - sigma*sigma / 2.0)*t + sigma*sqrt(t)*dZ);
}

template<typename T = double>
T monteCarloSim(const T & S0, const T & sigma, const T & r, double t, double K) {
    std::random_device rd;
    std::mt19937 rengine(rd());
    std::normal_distribution<> normal_dist(0, 1);
    T res = 0;
    double N = 100000;
    for (int i = 0; i < (int)N; i++)
        res += PayOff(BlackProcess(S0, sigma, r, t, normal_dist(rengine)), K);
    return res / N;
}


int main(int argc, char* argv[])
{
    
    size_t m = 3;
    vector<AD<double> > x(m);
    x[0] = 100; // Spot
    x[1] = 0.1; // implied Vola
    x[2] = 0.03; // risk free rate

    CppAD::Independent(x);
    
    vector<AD<double> > y(1);
    y[0] = monteCarloSim(x[0], x[1], x[2], 1., 110.);

    CppAD::ADFun<double> f(x, y);

    vector<double> jac(m); // Jacobian of f (m by n matrix)
    vector<double> xs(3);       // domain space vector
    xs[0] = 101; // argument value for derivative
    xs[1] = 0.1;
    xs[2] = 0.03;
    jac = f.Jacobian(xs);
    cout << "Delta " << jac[0] << endl;
    cout << "Vega " << jac[1] << endl;
    cout << "Rho " << jac[2] << endl;
    cout << "NPV" << monteCarloSim(100., 0.1, 0.03, 1., 110.) << endl;
    string input;
    cin >> input;
    return 0;
}

