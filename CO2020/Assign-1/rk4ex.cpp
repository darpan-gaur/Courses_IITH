#include <bits/stdc++.h>

using namespace std;



double dy1(double t,double y1,double y2){
    return 0; 
}

double dy2(double t,double y1,double y2){
    return 0;
}


double rk4(double a, double b,double x0,double x1,double h){
    int i;
    double k1,k2,k3,k4,ti,X,l1,l2,l3,l4,Y;
    ti = a;
    X = x0;
    Y = x1;
    while (abs(ti-b)>1e-6) {
        // cout << "Error :- " << 100*(x(ti)-X)/x(ti) << "\n";
        // printf("%.15f,%.15f,%.15f,%.15f\n",ti,x0,x(ti),abs(x0-x(ti)));
        k1 = h*dy1(ti,x0,x1);
        l1 = h*dy2(ti,x0,x1);
        k2 = h*dy1(ti+h/2,x0+k1/2,x1+l1/2);
        l2 = h*dy2(ti+h/2,x0+k1/2,x1+l1/2);
        k3 = h*dy1(ti+h/2,x0+k2/2,x1+l2/2);
        l3 = h*dy2(ti+h/2,x0+k2/2,x1+l2/2);
        k4 = h*dy1(ti+h,x0+k3,x1+l3);
        l4 = h*dy2(ti+h,x0+k3,x1+l3);
        X = x0 + (k1 + 2*k2 + 2*k3 + k4)/6;
        Y = x1 + (l1 + 2*l2 + 2*l3 + l4)/6;
        x0 = X;
        x1 = Y;
        ti+=h;
        // cout << k1 << " " << k2 << "\n";
        // cout << "t = " << ti << ", y = " << X << "\n";   
    }
    // printf("%.15f,%.15f,%.15f,%.15f\n",h,x0,x(ti),abs(x0-x(ti)));
    // cout << "t = " << ti << ", y = " << X << "\n";
    return X;
}

int main(){
    

    return 0;
}