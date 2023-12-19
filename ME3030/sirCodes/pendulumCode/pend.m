function zdot=pend(t,z)
global m k c L omega g A

xb=0;
yb=A*sin(omega*t);
xbd=0;
ybd=A*omega*cos(omega*t);

x=z(1);
y=z(2);
xd=z(3);
yd=z(4);

rm=[x y]';
rmd=[xd yd]';
rb=[xb yb]';
rbd=[xbd ybd]';
u=(rm-rb)/norm(rm-rb);
Fs=-k*(norm(rm-rb)-L)*u;
Fd=-c*(rmd-rbd);
F=Fs/m+Fd/m-[0 m*g]'/m;

zdot=[z(3);z(4);F];
