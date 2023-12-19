function zdot=rk42m(t,z)

global m g k L c kc mu

RHS=zeros(8,1);
s=z;
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
fc1=[0 0]';
fc2=[0 0]';
ffr1=[0 0]';
ffr2=[0 0]';

% if y1<0
%     fc1=[0 kc*abs(y1)]';
%     ffr1=[-mu*kc*abs(y1)*tanh(10*x1d) 0]';
% end
% 
% if y2<0
%     fc2=[0 kc*abs(y2)]';
%     ffr2=[-mu*kc*abs(y2)*tanh(10*x2d) 0]';
% end
am1=(-fg+fd+fs+fc1+ffr1)/m;
am2=(-fg-fd-fs+fc2+ffr2)/m;
RHS(5:8)=[am1;am2];
zdot=RHS;
