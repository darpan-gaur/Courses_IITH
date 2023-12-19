
function bc=solrk42m(vinit)

global m g k L c
m=1;
g=9.81;
k=1e3;
L=0.5;
c=5.0;
dt=0.01;
t=0:dt:2;
x=zeros(8,length(t));
x(:,1)=[0 0 0.5 0 vinit(1) vinit(2) vinit(3) vinit(4)]';
for i=1:1:length(t)-1
 
 t1=t(i);
 k1=rk42m(t1,x(:,i));
 k2=rk42m(t1+0.5*dt,x(:,i)+0.5*dt*k1) ; 
 k3=rk42m(t1+0.5*dt,x(:,i)+0.5*dt*k2);
 k4=rk42m(t1+dt,x(:,i)+dt*k3);
 x(:,i+1)=x(:,i)+(dt/6)*(k1+2*k2+2*k3+k4);
end







 x1=x(1,:);
 y1=x(2,:);
 
 x2=x(3,:);
 y2=x(4,:);
 
 bc=[x1(end) y1(end) x2(end) y2(end)]';

 
 x1d=x(5,:);
 y1d=x(6,:);
 x2d=x(7,:);
 y2d=x(8,:);





% for k=1:100:length(t)
% plot(x1(k), y1(k),'o') 
% hold on
% plot(x2(k), y2(k),'o')
% axis equal
% xlim([-100 100])
% ylim([-100 100])
% pause (0.1)
% hold off
% end
