    clc
clear all
m=1
g=0
k=1000
L=0.5
c=0
dt=0.00001;
t=0:dt:5;
z=zeros(8,length(t));
% z1=x1, z2=y1, z3=x2, z4=y2, z5=x1dot,z6=y1dot,z7=x2dot,z8=y2dot
z(:,1)=[0 0 0.5 0 0 1 0 -1]';
E=zeros(1,length(t));
for i=1:1:length(t)-1
 RHS=zeros(8,1);
 s=z(:,i);
 x1=s(1);
 y1=s(2);
 x2=s(3);
 y2=s(4);
 x1d=s(5);
 y1d=s(6);
 x2d=s(7);
 y2d=s(8);
 RHS(1)=s(5);
 RHS(2)=s(6);
 RHS(3)=s(7);
 RHS(4)=s(8);
 r1=[x1 y1]';
 r2=[x2 y2]';
 e=r2-r1;
 r1d=[x1d y1d]';
 r2d=[x2d y2d]';
 ed=r2d-r1d;
 fs=k*(norm(e,2)-L)*(e/norm(e,2));
 fd=c*ed;
 fg=[0 ; m*g];
 am1=(-fg+fd+fs)/m;
 am2=(-fg-fd-fs)/m;
 RHS(5:8)=[am1;am2];
 E(i+1)=0.5*m*(x1d^2+y1d^2+x2d^2+y2d^2)+0.5*k*(norm(e,2)-L)^2;
 z(:,i+1)=z(:,i)+dt*RHS  ;     
end
plot(t(2:end),E(2:end))

 x1=z(1,:);
 y1=z(2,:);
 
 x2=z(3,:);
 y2=z(4,:);
 
 x1d=z(5,:);
 y1d=z(6,:);
 x2d=z(7,:);
 y2d=z(8,:);

for k=1:1000:length(t)
    
   plot(x1(k), y1(k),'o') 
   hold on
   plot(x2(k), y2(k),'o') 

xlim([-2 2])
ylim([-2 2])
axis equal 
pause (0.1)
hold off

end
