function zdot=MagicCar(t,z)
global m J a A omega c

M=diag([m m J]);

alpha =A*sin(omega*t);
alphad =A*omega*cos(omega*t);
alphadd =-A*omega^2*sin(omega*t);


F=[0 0 0]';

xcg=z(1);
ycg=z(2);
theta=z(3);

xdcg=z(4);
ydcg=z(5);
thetad=z(6);

rcg=[xcg ycg]';
rdcg=[xdcg ydcg]';

Rw=[cos(theta) -sin(theta); sin(theta) cos(theta)];
Q=[-sin(theta) -cos(theta); cos(theta) -sin(theta)];
Rdw=thetad*Q;
Qd=thetad*[-cos(theta) sin(theta); -sin(theta) -cos(theta)];
%Rddw=thetadd*Q+thetad*Qd

rw=rcg+Rw*[0 -a]';
rdw=rdcg+Rdw*[0 -a]';
%rddw=rddcg+thetadd*Q*[0 -a]'+thetad*Qd*[0 -a]'

rtw=Rw*[1 0]';
rdtw=Rdw*[1 0]';
Rhs1=-rdtw'*rdw-thetad*rdtw'*(Qd*[0 -a]');
Lhs1=[rtw' rtw'*Q*[0 -a]'];

Rp=[cos(theta+alpha) -sin(theta+alpha); sin(theta+alpha) cos(theta+alpha)];
S=[-sin(theta+alpha) -cos(theta+alpha); cos(theta+alpha) -sin(theta+alpha)];
Rdp=(thetad+alphad)*S;
Sd=(thetad+alphad)*[-cos(theta+alpha) sin(theta+alpha); -sin(theta+alpha) -cos(theta+alpha)];

% %Rddp=thetadd*S+alphadd*S+thetad*Sd+alphad*Sd
% 
rp=rcg+Rw*[0 a]'+Rp*[0 -c]';
rdp=rdcg+Rdw*[0 a]'+Rdp*[0 -c]';
%rddp=rddcg+(thetadd*Q+thetad*Qd)*[0 a]'...
%         +(thetadd*S+alphadd*S+thetad*Sd+alphad*Sd)*[0 -c]';

rtp=Rp*[1 0]';
rdtp=Rdp*[1 0]';

Lhs2=[rtp' rtp'*Q*[0 a]'+rtp'*S*[0 -c]'];

Rhs2=-rtp'*thetad*Qd*[0 a]'...
     -rtp'*(alphadd*S+thetad*Sd+alphad*Sd)*[0 -c]'-rdtp'*rdp;
C1=rtw'*rdw;
C2=rtp'*rdp;

U=[Lhs1; Lhs2];
v=[Rhs1-10*C1;Rhs2-10*C2];
 
[C1 ; C2]
% v=[Rhs1+Rhs2; Rhs3+Rhs4]
% E1=[Rhs1+Rhs2-v]
% E2=[Lhs1;U]


acc=M\F+(M^(-0.5))*pinv(U*(M^(-0.5)))*(v-U*(M\F));
zdot=[z(4) z(5) z(6) acc']';