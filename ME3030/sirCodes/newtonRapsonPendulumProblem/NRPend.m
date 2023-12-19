
clc
clear all
global alpha
alpha=3;
beta=5;
db=1e-3;
tspan=0:0.01:2;
for i=1:1:100
[t,z]=ode45(@pendbvp,tspan,[0 beta]); 
[t,zb]=ode45(@pendbvp,tspan,[0 beta+db]);
theta=z(:,1); % solution for nominal value of beta
thetab=zb(:,1); % solution for perturbed value of beta
f=theta(end)-pi/2 % error function for nominal value of beta
fb=thetab(end)-pi/2; % error function for perturbed value of beta
fd=(fb-f)/db; % calculation of derivative using finite difference formula
beta=beta-f/fd; % updating beta
end
plot(t,theta)