% pendulum problem
% solved using matlab integrator
clc
clear all
global m k c L omega g A
A=0.01
m=1
k=1e6
c=0.2
L=1
g=10
omega=1.54*(sqrt(2*g*L)/A);

tspan=0:0.01:10;
init=[L*sin(pi/20) L*cos(pi/20) 0 0]
options=odeset('AbsTol',1e-6,'RelTol',1e-6)
[t,z]=ode45(@pend,tspan,init,options)

x=z(:,1);
y=z(:,2);
for i=1:5:length(t)
plot([0 x(i)],[A*sin(omega*t(i)) y(i)],'o-')
axis equal
xlim([-2 2])
ylim([-2 2])
pause(0.1)
hold off
end
% e=abs(y-A*sin(omega*t))
% plot(t,e)