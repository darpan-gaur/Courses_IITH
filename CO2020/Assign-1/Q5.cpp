#include <bits/stdc++.h>

using namespace std;

// differential equation
double dx_dt(double t, double x){
	return (t*x*x-x)/t;
}

// Exact Solution
double x(double t){
	return -1/(t*log(2*t));
}

// RK-2
double rk2(double a, double b,double x0,double h){
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
		ti= ti + h;	
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
	return X;
}

// Adam Bashforth Moulton 2nd Order
void admb2(double a,double b,double x0,double h){
	vector<double> y(2);
	vector<double> v(2);
	double t0 = a,Y;
	int i;
	y[0] = x0;
	v[0] = dx_dt(t0,y[0]);
	// printf("%.15f,%.15f,%.15f,%.15f\n",t0,y[0],x(t0),abs(y[0]-x(t0)));
	for (i=0;i<1;i++) {
		y[i+1] = rk2(t0,t0+h,y[i],h);
		v[i+1] = dx_dt(t0+h,y[i+1]);
		t0 += h;
	}
	// printf("%.15f,%.15f,%.15f,%.15f\n",t0,y[1],x(t0),abs(y[1]-x(t0)));

	while (abs(t0-b)>1e-6){
		Y = y[1] + (h/2)*(3*v[0]-2*v[1]);
		for (i=0;i<1;i++) {
			v[i] = v[i+1];
		}
		v[1] = dx_dt(t0+h,Y);
		Y = y[1] + (h/2)*(v[0]+v[1]);
		v[1] = dx_dt(t0+h,Y);
		t0+=h;
		// printf("%.15f,%.15f,%.15f,%.15f\n",t0,Y,x(t0),abs(Y-x(t0)));
		y[1] = Y;
		
		
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",t0,Y,x(t0),abs(Y-x(t0)));
}

int main(){
    cout << "t,Approximate Solution,Exact Solution,Absolute Error\n";

    double a,b,h;
    // bounds of t :- a <= t <= b
    // h :- step size

    a = 1;
    b = 5;
    h = 0.001;

    // Print t, Approximate Solution, Exact Solution, Absolute Error
	rk2(a,b,x(1),h);

	// Print t, Approximate Solution, Exact Solution, Absolute Error
	admb2(a,b,x(1),h);
    return 0;
}