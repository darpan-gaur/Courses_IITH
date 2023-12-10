#include <bits/stdc++.h>

using namespace std;

// differential Equaiton
double f(double t, double x){
	return 4*x/t+(t*t*t*t)*(exp(t));
}

// Exact Solution
double x(double t){
	return (t*t*t*t)*(exp(t)-exp(1));
}

// RK-4
double rk4(double a, double b,double x0,double h){
	int i;
	double k1,k2,k3,k4,ti,X;
	ti = a;
	X = x0;
	while (abs(ti-b)>1e-6) {
		// printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
		k1 = h*f(ti,x0);
		k2 = h*f(ti+h/2,x0+k1/2);
		k3 = h*f(ti+h/2,x0+k2/2);
		k4 = h*f(ti+h,x0+k3);
		X = x0 + (k1 + 2*k2 + 2*k3 +k4)/6;
		x0 = X;
		ti+=h;
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
	return X;
}

// Adam Bashforth Moulton 4th Order
void admb4(double a,double b,double x0,double h){
	vector<double> y(4);
	vector<double> v(4);
	double t0 = a,Y;
	int i;
	y[0] = x0;
	v[0] = f(t0,y[0]);
	// printf("%.15f,%.15f,%.15f,%.15f\n",t0,y[0],x(t0),abs(y[0]-x(t0)));
	for (i=0;i<3;i++) {
		y[i+1] = rk4(t0,t0+h,y[i],h);
		v[i+1] = f(t0+h,y[i+1]);
		t0 += h;
		// printf("%.15f,%.15f,%.15f,%.15f\n",t0,y[i],x(t0),abs(y[i]-x(t0)));
	}

	while (abs(t0-b)>1e-6){
		Y = y[3] + (h/24)*(-9*v[0]+37*v[1]-59*v[2]+55*v[3]);
		for (i=0;i<3;i++) {
			v[i] = v[i+1];
		}
		v[3] = f(t0+h,Y);
		Y = y[3] + (h/24)*(v[0]-5*v[1]+19*v[2]+9*v[3]);
		v[3] = f(t0+h,Y);
		
		t0+=h;
		// printf("%.15f,%.15f,%.15f,%.15f\n",t0,Y,x(t0),abs(Y-x(t0)));
		y[3] = Y;
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",t0,Y,x(t0),abs(Y-x(t0)));
}

int main(){
	cout << "t,Approximate Solution,Exact Solution,Absolute Error\n";

	double a,b,x0,h;
    // bounds of t :- a <= t <= b
    // x0 :- intial value
    // h :- step size
    a = 1;
    b = 2;
    x0 = 0;
    h = 0.001;

    // Print t, Approximate Solution, Exact Solution, Absolute Error
	rk4(a,b,x0,h);

	// Print t, Approximate Solution, Exact Solution, Absolute Error
    admb4(a,b,x0,h);

    return 0;
}