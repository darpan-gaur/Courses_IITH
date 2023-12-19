% one mass, spring and damper
% r1 oscillating with c + Asinwt
clc
clear all
m=10;
k=5000;
c=100;
L=1;
dt=0.001;
t=0:dt:30;
omega=sqrt(k/m)
r1=0.5+0.2*sin(omega*t);
r1dot=0.2*omega*cos(omega*t);
x1=zeros(1,length(t));
x2=zeros(1,length(t));
x1(1)=r1(1)+L;
x2(1)=0;

for i=1:1:length(t)-1
x1(i+1)=x1(i)+dt*x2(i);
x2(i+1)=x2(i)+dt*(1/m)*(-c*x2(i)-k*x1(i)+k*(r1(i)+L)+c*r1dot(i));
end

r2=x1
for i=1:400:length(t)
hold off
plot([r1(i) r2(i)],[0 0],'o','markersize',20)
xlim([0 20])
pause(0.1)
hold off
end

figure(2)
plot(t,x1,'b')
hold on
plot(t,r1,'r')
