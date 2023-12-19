clc
clear all
global m J a  A omega c

m=1;
J=1;
a=0.25;
b=0.5
A=pi/10
c=0.1
omega=4
init=[0 0 0 0 0 0];
tspan=0:0.1:20;
options=odeset('Reltol',1e-8,'AbsTol',1e-8)
[t,z]=ode45(@MagicCar,tspan,init,options);
xcg=z(:,1);
ycg=z(:,2);
theta=z(:,3);
xdcg=z(:,4);
ydcg=z(:,5);
thetad=z(:,6);

for i=1:1:length(t)
   alpha =A*sin(omega*t(i));
   rcg=[xcg(i) ycg(i)]';
   R=[cos(theta(i)) -sin(theta(i)); sin(theta(i)) cos(theta(i))];
   beta=theta(i)+alpha   ;
   Rp=[cos(beta) -sin(beta); sin(beta) cos(beta)];
   r1=rcg+R*[a b]';
   r2=rcg+R*[-a b]';
   r3=rcg+R*[-a -b]';
   r4=rcg+R*[a -b]';
   plot([r1(1) r2(1) r3(1) r4(1) r1(1)],[r1(2) r2(2) r3(2) r4(2) r1(2)]);
   hold on
   rw=rcg+R*[0 -a]';
   rc=rcg+R*[0 a]';
   rp=rc+Rp*[0 -2*c]';
   rtp=rc+Rp*[2*c 0]';
   plot(rw(1),rw(2),'s')
   hold on
plot([rc(1) rp(1)],[rc(2) rp(2)],'-','linewidth',2)
   hold on
plot([rc(1) rtp(1)],[rc(2) rtp(2)],'-','linewidth',2)
hold on
axis equal
% xlim([-1 1])
% ylim([-4 4])
   pause(0.1)
   
   hold off
end
