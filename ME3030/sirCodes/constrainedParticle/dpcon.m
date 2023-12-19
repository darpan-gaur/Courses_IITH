
function zdot=dpcon(t,z)
global m1 m2 L1 L2 g alpha beta
RHS=zeros(8,1);
s=z(1:8);
x1=s(1);
y1=s(2);
x2=s(3);
y2=s(4);
x1d=s(5);
y1d=s(6);
x2d=s(7);
y2d=s(8);

RHS(1)=s(5);
RHS(2)=s(6);
RHS(3)=s(7);
RHS(4)=s(8);


M=[m1 0 0 0; 0 m1 0 0 ; 0 0 m2 0 ; 0 0 0 m2];
f=[0 -m1*g 0 -m2*g]';
C1=x1^2+y1^2-L1^2;
C2=(x1-x2)^2+(y1-y2)^2-L2^2;
C3=y2
C=[C1 C2 C3]';
q=[x1 y1 x2 y2]';
qd=[x1d y1d x2d y2d]';
J=[2*x1 2*y1 0 0 ; 2*(x1-x2) 2*(y1-y2) -2*(x1-x2) -2*(y1-y2); 0 0 0 1];
Jd=[2*x1d 2*y1d 0 0 ; 2*(x1d-x2d) 2*(y1d-y2d) -2*(x1d-x2d) -2*(y1d-y2d); 0 0 0 0];
D=inv(J*inv(M)*J');
lam=D*(-Jd*qd-alpha*J*qd-beta*C-J*inv(M)*f);
fc=J'*lam;
acc=inv(M)*(f+fc)
RHS(5:8)=acc;
zdot=RHS;

