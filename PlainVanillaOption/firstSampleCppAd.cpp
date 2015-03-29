#include <cppad/cppad.hpp>
#include <vector>
#include <iostream>

using namespace std;

using CppAD::AD;

template<class T = double>
T fun(const T& x1, const T& x2){
    return x1*x1+2*x2+3*x1*x2+1+3*x2*x2*x2;
}

int main(){
    vector<AD<double> > x(2);
    x[0] = -3;
    x[1] = 5;
    CppAD::Independent(x);
    vector<AD<double> > y(1);
    y[0] = fun(x[0],x[1]);
    CppAD::ADFun<double> f(x,y);
    cout << "fun(x) = " << y[0] << endl;
    vector<double> dx(2,0);
    dx[0] = 1;
    cout << "dfun(x) / dx_1 = " << f.Forward(1,dx)[0] << endl;
    vector<double> ddx(2,0);
    cout << "ddfun(x) / dx1 dx1 = " << 2.0 * f.Forward(2, ddx)[0] <<endl;
    dx[0] = 0;
    dx[1] = 1;
    cout << "dfun(x) / dx_2 = " << f.Forward(1,dx)[0] << endl;
    cout << "ddfun(x) / dx2 dx2 = " << 2.0 * f.Forward(2, ddx)[0] <<endl;
   
    cout << "Backward methoden " << endl;
    vector<double> w(1,1);
    vector<double> dFun(2,0);
    dFun = f.Reverse(1, w);
    cout << "dfun(x) / dx1 = " << dFun[0] << endl;
    cout << "dfun(x) / dx2 = " << dFun[1] << endl;
    dFun.resize(2*2);
    dFun = f.Reverse(2, w);
    cout << "dfun(x) / dx1 = " << dFun[0] << endl;
    dx[0] = 1;
    dx[1] = 0;
    f.Forward(1,dx)[0];
    dFun = f.Reverse(2, w);
    cout << "dfun(x) / dx1 dx1 = " << dFun[1] << endl;
    cout << "dfun(x) / dx2 = " << dFun[2] << endl;
    cout << "dfun(x) / dx2 dx2 = " << dFun[3] << endl;

    
}
