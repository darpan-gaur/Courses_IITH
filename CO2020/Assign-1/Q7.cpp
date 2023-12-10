#include <bits/stdc++.h>

using namespace std;

// d^2(x)/dt^2
double dy1(double t,double u0,double u1){
    return (1-u0*u0)*u1-u0;
}

//RK-4
void rk4(double a, double b,double u0,double u1,double h){
    int i;
    double k1,k2,k3,k4,ti,X,l1,l2,l3,l4,Y;
    ti = a;
    while (abs(ti-b)>1e-6) {
        // Print time and x value at that time
        printf("%.15f,%.15f\n",ti,u0);

        k1 = h*dy1(ti,u0,u1);
        l1 = h*u1;
        k2 = h*dy1(ti+h/2,u0+l1/2,u1+k1/2);
        l2 = h*(u1+k1/2);
        k3 = h*dy1(ti+h/2,u0+l2/2,u1+k2/2);
        l3 = h*(u1+k2/2);
        k4 = h*dy1(ti+h,u0+l3,u1+k3);
        l4 = h*(u1+k3);
        u1 = u1 + (k1 + 2*k2 + 2*k3 + k4)/6;
        u0 = u0 + (l1 + 2*l2 + 2*l3 + l4)/6;
        ti+=h;
  
    }
    // Print time and x value at that time
    printf("%.15f,%.15f\n",ti,u0);
}

int main(){
    cout << "t,x\n";

    double a,b,x0,dx_0,h;
    // bounds of t :- a <= t <= b
    // x0  :- initial value of x at t=a
    // dx0 :- initail value of derivative of x at t=a
    // h   :- step size
    a = 0;
    b = 30;
    x0 = 0.5;
    dx_0 = 0.1;
    h = 0.1;

    // Print time and x value at that time for a<= t <=b
    rk4(a,b,x0,dx_0,h);
    return 0;
}