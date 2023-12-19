% solving thetaddot+alpha*sin(theta)=T(t)=A*sin(omega*t)
clc
clear all
global alpha omega A

alpha=2;
A=3;
omega=5;

dt=1e-3;
t=0:dt:1;
x1=zeros(1,length(t));
x2=zeros(1,length(t));


x1(1)=pi/4;
x2(1)=1;

for k=1:1:length(t)-1
k
T=t(k+1);
z1=x1(k);
z2=x2(k);
f1=z1-x1(k)-dt*z2;
f2=z2-x2(k)+dt*(alpha*sin(z1))-dt*A*sin(omega*T);

while max(abs([f1,f2]))>1e-6   
f=[f1 f2]';
J=[1 -dt; dt*alpha*cos(z1) 1]
r=inv(J)*f;
z1=z1-r(1);
z2=z2-r(2);
f1=z1-x1(k)-dt*z2;
f2=z2-x2(k)+dt*alpha*sin(z1)-dt*A*sin(omega*T);
end
x1(k+1)=z1;
x2(k+1)=z2;
end
plot(t,x1)
hold on
plot(t,x2)

% validation with ode45 (sanity check)
[t,y]=ode45(@pendval,t,[x1(1) x2(1)]);
hold on
plot(t,y(:,2))


