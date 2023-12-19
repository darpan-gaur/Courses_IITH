
guess= [5.0; 10.0; -1.0; 5.0];

for i=1:1:100
    i
dv=1e-3; % Finite difference derivative paraemter
p=dv*eye(4,4);% pertubation
n=solrk42m(guess); % nominal solution
pvx1=solrk42m(guess+p(:,1)); % solution due to perturbation in vx1 
pvy1=solrk42m(guess+p(:,2));% solution due to perturbation in vy1
pvx2=solrk42m(guess+p(:,3));% solution due to perturbation in vx2
pvy2=solrk42m(guess+p(:,4));% solution due to perturbation in vy2
J=(1/dv)*[pvx1-n  pvy1-n  pvx2-n  pvy2-n ]; % Jacobian
f=n-[ 1 1 1 1.5]' % error function
if norm(f)<1e-6 
    break
end
guess=guess-J\f; % NR update
end
format longg
guess
