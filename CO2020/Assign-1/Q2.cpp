#include <bits/stdc++.h>

using namespace std;

// Initializing constants globally
double k = 20;		// k :- environmental carrying capacity
double r = 0.4;		// r :- natural growth rate of the population

// differntial equation
double f(double x,double t) {
	return r*x*(1-x/k)-x*x/(1+x*x);
}

void Euler(double a, double x0,double h){
	int i;
	double t0 = a,x1;
	for (i=1;;i++) {
		// printf("%.15f,%.15f\n",t0,x0);
		x1 = x0 + h*f(x0,t0);

		// Break when eventual population level is reached
		if (abs(x1-x0) < 1e-12) break;
		x0 = x1; 
		t0 += h;
		
	}
	// Print time and eventual population level
	printf("%.15f,%.15f\n",t0,x0);

}

int main(){
	cout << "t,x\n";

	double a,x0,h;
	// a  :- initial time value
	// x0 :- inital population value
	// h  :- Step size
	a = 0;
	x0 = 2.44;
	h = 0.00001;

	// Print time and eventual population level
    Euler(a,x0,h);

    x0 = 2.4;
    // Print time and eventual population level
    Euler(a,x0,h);
    return 0;
}