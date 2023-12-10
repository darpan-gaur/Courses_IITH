#include <bits/stdc++.h>

using namespace std;

// Initializing constants globally
double d=0.002;					// plate thickness
double rho=7900;				// Density of AISI 304 stainless steel
double sigma=5.670374419e-8;	// Stefan boltzmann constant
double Tf=1500;					// Temperature of furnance

// Calculates Specific heat at given temperature
double Cp(double T){
	return 0.162*T+446.47;
}

// derivate of specifict heat at given temperature
double dCp_dT (double T){
	return 0.162;
}

// differential equation
double f(double t,double T) {
	return 2*sigma*(Tf*Tf*Tf*Tf-T*T*T*T)/(rho*d*(Cp(T)+T*(dCp_dT(T))));
}

void rk4(double t0,double T0,double h){
	int i;
	double k1,k2,k3,k4,ti,T;
	ti = t0;
	while (abs(f(ti,T))>1e-6) {
		// printf("%.15f,%.15f\n",ti,T0);
		k1 = h*f(ti,T0);
		k2 = h*f(ti+h/2,T0+k1/2);
		k3 = h*f(ti+h/2,T0+k2/2);
		k4 = h*f(ti+h,T0+k3);
		T = T0 + (k1 + 2*k2 + 2*k3 + k4)/6;		
		T0 = T;
		ti+=h;
	}
	// Print time to reach thermal equilibrim (ti) and Temperature (T0)
	printf("%.15f,%.15f\n",ti,T0);
}

int main(){
	cout << "t,T\n";
	
	// t0 :- starting time
	// T0 :- starting temperature
	// h  :- step size
	double t0,T0,h;
	t0 = 0;
	T0 = 300;
	h  = 0.0001;

	// Print time to reach thermal equilibrim (t) and Temperature (T)
   	rk4(t0,T0,h);
    return 0;
}