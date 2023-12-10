#include <bits/stdc++.h>

using namespace std;

// differential eqaution
double f(double x,double t) {
	return -(1+t+t*t)-(2*t+1)*x-x*x;
}

// Part (a) Exact Solution
double x1(double t) {
	return -t-1/(exp(t)+1);
}

// Part (b) Exact Solution
double x2(double t){
	return -t-1;
}

void Euler(double a,double b, double x0,double h,double (*x)(double)){
	int i;
	double t0 = a;
	while (t0<b) {
		// printf("%.15f,%.15f,%.15f,%.15f\n",t0,x0,x(t0),abs(x0-x(t0)));
		x0 = x0 + h*f(x0,t0);
		t0 += h;
	}
	// Print t, Approximate Solution, Exact Solution, Absolute Error
	printf("%.15f,%.15f,%.15f,%.15f\n",t0,x0,x(t0),abs(x0-x(t0)));
}

int main(){
	cout << "t,Approximate Solution,Exact Solution,Absolute Error\n";
	
	// bounds of t :- a <= t <= b
	// x0 :- Initial vale x(0)
	// h  :- step size
	double a,b,x0,h,t0;

	// Part (a)
	a  = 0;
	b  = 3;
	x0 = -0.5;
	h  = 0.5;
	// At t=b:- print t, Approximate Solution, Exact Solution, Absolute Error
	Euler(a,b,x0,h,&x1);
	
	// Part (b)
	a = 0;
	b = 3;
	x0 = -1;
	h = 0.5;
	// At t=b:- print t, Approximate Solution, Exact Solution, Absolute Error
	Euler(a,b,x0,h,&x2);
    return 0;
}