clc
clear all
global alpha beta a b c m gr
alpha=10;
beta=5;
m=1;
a=0.6*3;
b=0.9*3;
c=1*3;
gr=0;

Ixx=4*m*(a^2+c^2)
Iyy=4*m*(b^2+c^2)
Izz=4*m*(a^2+b^2)

% [omegax omegay omegaz] used in the initial condition
omega=[4 0.01 0.01]'
p1i=[0 -a 0]';  v1i=skew(omega)*p1i;
p2i=[b 0 0]';   v2i=skew(omega)*p2i;
p3i=[0 a 0]';   v3i=skew(omega)*p3i;
p4i=[-b 0 0]';  v4i=skew(omega)*p4i;
p5i=[0 0 c]';   v5i=skew(omega)*p5i;
p6i=[0 0 -c]';  v6i=skew(omega)*p6i;



init=[p1i ; p2i;  p3i;  p4i;  p5i;  p6i;  v1i;  v2i ; v3i ; v4i;  v5i;  v6i]
tspan=0:0.1:100;
options=odeset('AbsTol',1e-8,'RelTol',1e-8)
[t,z]=ode15s(@stabrb6,tspan,init,options);ss
x1=z(:,1); y1=z(:,2); z1=z(:,3);
x2=z(:,4); y2=z(:,5); z2=z(:,6);
x3=z(:,7); y3=z(:,8); z3=z(:,9);
x4=z(:,10); y4=z(:,11); z4=z(:,12);

x5=z(:,13); y5=z(:,14); z5=z(:,15);
x6=z(:,16); y6=z(:,17); z6=z(:,18);


for i=1:1:length(t)
    X=[x1(i) x2(i) x3(i) x4(i)  x5(i) x6(i) ];
    Y=[y1(i) y2(i) y3(i) y4(i)  y5(i) y6(i) ];
    Z=[z1(i) z2(i) z3(i) z4(i)  z5(i) z6(i) ];
       
    fill3([x1(i) x2(i) x5(i)], [y1(i) y2(i) y5(i)], [z1(i) z2(i) z5(i)],[0.3010 0.7450 0.9330]); hold on
    fill3([x2(i) x3(i) x5(i)], [y2(i) y3(i) y5(i)], [z2(i) z3(i) z5(i)],[0.4940 0.1840 0.5560]); hold on 
    fill3([x3(i) x4(i) x5(i)], [y3(i) y4(i) y5(i)], [z3(i) z4(i) z5(i)],[0.8500 0.3250 0.0980]); hold on 
    fill3([x1(i) x4(i) x5(i)], [y1(i) y4(i) y5(i)], [z1(i) z4(i) z5(i)],[0 0.7 0.7]); hold on
    
    fill3([x1(i) x2(i) x6(i)], [y1(i) y2(i) y6(i)], [z1(i) z2(i) z6(i)],[0 0.4470 0.7410]); hold on
    fill3([x2(i) x3(i) x6(i)], [y2(i) y3(i) y6(i)], [z2(i) z3(i) z6(i)],[0.9290 0.6940 0.1250]); hold on 
    fill3([x3(i) x4(i) x6(i)], [y3(i) y4(i) y6(i)], [z3(i) z4(i) z6(i)],[0.6350 0.0780 0.1840]); hold on 
    fill3([x1(i) x4(i) x6(i)], [y1(i) y4(i) y6(i)], [z1(i) z4(i) z6(i)],'k'); hold on
    axis equal 
    
    plot3([0 0 2*x5(i)],[0 0 2*y5(i)],[0 0 2*z5(i)],'b','linewidth',2); hold on
    plot3([0 0 2*x2(i)],[0 0 2*y2(i)],[0 0 2*z2(i)],'k','linewidth',2); hold on
    plot3([0 0 2*x3(i)],[0 0 2*y3(i)],[0 0 2*z3(i)],'r','linewidth',2); hold on
    
    xlim([-2*c 2*c])
    ylim([-2*c 2*c])
    zlim([-2*c 2*c])
    
    pause(0.02);
    hold off
end

    figure(2)
    vel=z(1:length(t),19:36);
    plot(t,sum((vel.^2)'))