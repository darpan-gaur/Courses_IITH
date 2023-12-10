#include <bits/stdc++.h>

using namespace std;

// differential eqaution
double dx_dt(double t, double x){
	return (7*t*t)-4*x/t;
}

// Exact Solution
double x(double t){
	return (t*t*t)+1/(t*t*t*t);
}

// RK-2
void rk2(double a, double b,double x0,double h){
	int i;
	double k1,k2,ti,X;
	ti = a;
	X = x0;
	while (abs(ti-b)>1e-6) {
		// printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
		k1 = h*dx_dt(ti,x0);
		k2 = h*dx_dt(ti+h,x0+k1);
		X = x0 + (k1 + k2)/2;
		x0 = X;
		ti+=h;	
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
	
	// Print percentage error
	printf("Percentage Error :- %.15f\n",100*abs(x0-x(ti))/abs(x(ti)));
}

// RK-4
void rk4(double a, double b,double x0,double h){
	int i;
	double k1,k2,k3,k4,ti,X;
	ti = a;
	X = x0;
	while (abs(ti-b)>1e-6) {
		// printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
		k1 = h*dx_dt(ti,x0);
		k2 = h*dx_dt(ti+h/2,x0+k1/2);
		k3 = h*dx_dt(ti+h/2,x0+k2/2);
		k4 = h*dx_dt(ti+h,x0+k3);
		X = x0 + (k1 + 2*k2 + 2*k3 +k4)/6;
		x0 = X;
		ti+=h;
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));

	// Print percentage error
	printf("Percentage Error :- %.15f\n",100*abs(x0-x(ti))/abs(x(ti)));
}

int main(){
	cout << "t,Approximate Solution,Exact Solution,Absolute Error\n";
	
	double a,b,x0,h;
	// bounds of t :- a <= t <= b
	// x0 :- intial value
	// h  :- step size

	a = 1;
	b = 6;
	x0 = 2;
	h = 0.01;

	// At t=b:- print t, Approximate Solution, Exact Solution, Absolute Error
	// Print percentage error
	rk2(a,b,x0,h);
	
	// At t=b:- print t, Approximate Solution, Exact Solution, Absolute Error
	// Print percentage error
	rk4(a,b,x0,h);
    return 0;
}