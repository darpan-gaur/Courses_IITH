clc
clear all
global m J a b g A omega

m=1;
J=1;
a=0.25;
b=0.25;
g=1;
A=0.1
omega=40

init=[1 1+b+A pi+(pi/4) 1e-3 1e-3 0];
tspan=0:0.1:40;
options=odeset('Reltol',1e-8,'AbsTol',1e-8)
[t,z]=ode15s(@pinsq,tspan,init,options);
xcg=z(:,1);
ycg=z(:,2);
theta=z(:,3);
xdcg=z(:,4);
ydcg=z(:,5);
thetad=z(:,6);
for i=1:1:length(t)
   rcg=[xcg(i) ycg(i)]';
   R=[cos(theta(i)) -sin(theta(i)); sin(theta(i)) cos(theta(i))];
   r1=rcg+R*[a b]';
   r2=rcg+R*[-a b]';
   r3=rcg+R*[-a -b]';
   r4=rcg+R*[a -b]';
   plot([r1(1) r2(1) r3(1) r4(1) r1(1)],[r1(2) r2(2) r3(2) r4(2) r1(2)],'o-');
   axis equal
   xlim([0 2])
   ylim([0 2])
   pause(0.1)
   
   hold off
end

C=zeros(1,length(t))
Cd=zeros(1,length(t))

for i=1:1:length(t)
    i
xc=1;
yc=1+A*cos(omega*t(i))

xcd=0;
ycd=-A*omega*sin(omega*t(i))

rcg=[xcg(i) ycg(i)]';
vcg=[xdcg(i) ydcg(i)]';

rc=[xc yc]';
rcd=[xcd ycd]'

R=[cos(theta(i)) -sin(theta(i)) ; sin(theta(i)) cos(theta(i))];
Rd=thetad(i)*[-sin(theta(i)) -cos(theta(i)) ; cos(theta(i)) -sin(theta(i))];

C(i)=max(abs(rcg+R*[a b]'-rc))
Cd(i)=max(abs(vcg+Rd*[a b]'-rcd))

end
figure(2)
plot(t,C)
figure(3)
hold on
plot(t,Cd)
