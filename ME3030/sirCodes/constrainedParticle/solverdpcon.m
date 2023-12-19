
global m1 m2 L1 L2 g alpha beta
 m1=1 ;
 m2=0.5;
 L1=1;
 L2=2;
 g=10;
 alpha=10;
 beta=10;
init=[L1 0 L1+L2 0 0 0 0 0 ]
tspan=0:0.01:10;
options=odeset('AbsTol',1e-6,'RelTol',1e-6)
[t,z]=ode45(@dpcon,tspan,init,options)
x1=z(:,1);
y1=z(:,2);
x2=z(:,3);
y2=z(:,4);
x1d=z(:,5);
y1d=z(:,6);
x2d=z(:,7);
y2d=z(:,8);
for i=1:10:length(t)
    plot([0 x1(i) x2(i)],[0 y1(i) y2(i)],'o-');
   axis equal
   ylim([-3 3])
   xlim([-3 3])
   
    pause(0.1);
    hold off
end
figure(2)
C1=x1.^2+y1.^2-L1^2;
C2=(x1-x2).^2+(y1-y2).^2-L2^2;
C3=y2
plot(t,C1,t,C2,t,C3)
figure(3)
KE=0.5*(m1*x1d.^2+m1*y1d.^2+m2*x2d.^2+m2*y2d.^2)
PE=m1*g*y1+m2*g*y2
plot(t,KE+PE)