function zdot=pinsq(t,z)
global m J a b g A omega

xc=1;
yc=1+A*cos(omega*t);

xcd=0;
ycd=-A*sin(omega*t);

xcdd=0;
ycdd=-A*omega*omega*cos(omega*t);



M=diag([m m J]);
F=[0 -m*g 0]';
xcg=z(1);
ycg=z(2);
theta=z(3);

xdcg=z(4);
ydcg=z(5);
thetad=z(6);

rcg=[z(1) z(2)]';
vcg=[z(4) z(5)]';

rc=[xc yc]';
rcd=[xcd ycd]';

R=[cos(theta) -sin(theta) ; sin(theta) cos(theta)];
Rd=thetad*[-sin(theta) -cos(theta) ; cos(theta) -sin(theta)];

C=rcg+R*[a b]'-rc;
Cd=vcg+Rd*[a b]'-rcd;

E1=max(abs(C));
E2=max(abs(Cd));


U=[1 0 -a*sin(theta)-b*cos(theta); 0 1 a*cos(theta)-b*sin(theta)];
v=[xcdd ycdd]'+thetad^2*[a*cos(theta)-b*sin(theta) a*sin(theta)+b*cos(theta)]'-10*C-1*Cd;

acc=M\F+(M^(-0.5))*pinv(U*(M^(-0.5)))*(v-U*(M\F));

zdot=[z(4) z(5) z(6) acc']';