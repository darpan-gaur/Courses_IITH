
clc
clear all
% thetaddot+alpha*sin(theta)=0
% We are finding final time and initial velocity to match the 
%boundary conditions on angle and velocity at final time
global alpha

alpha=3;

% initial guess for velocity and final time 
beta=10;
T=1


db=1e-3;
dT=1e-2

mu=pi/2% final angle


for i=1:1:400

i
tspan=0:0.01:T;
tspandT=0:0.01:T+dT;    
    
[t,z]=ode45(@pendbvp,tspan,[0 beta]); 
[t,zb]=ode45(@pendbvp,tspan,[0 beta+db]);
[t,zT]=ode45(@pendbvp,tspandT,[0 beta]);



theta=z(:,1); % solution for nominal value of beta
thetadot=z(:,2);

thetab=zb(:,1); % solution for perturbed value of beta
thetaT=zT(:,1); % solution for perturbed value of T

thetadotb=zb(:,2); % velcity for perturbed value of beta
thetadotT=zT(:,2); % velcity for perturbed value of T

f1=theta(end)-mu; % functions of beta and T
f2=thetadot(end); % functions of beta and T

f=[f1 f2 ]';

f1b=thetab(end)-mu;
f1T=thetaT(end)-mu;

f2b=thetadotb(end);
f2T=thetadotT(end);

J11=(f1b-f1)/db;
J12=(f1T-f1)/dT;
J21=(f2b-f2)/db;
J22=(f2T-f2)/dT;
J=[J11 J12 ; J21 J22];

r=[beta T]'-inv(J)*f;

beta=r(1);
T=r(2);
if max(abs(f))<1e-2
    break
end

end
plot(tspan,theta)
hold on
plot(tspan,thetadot)
