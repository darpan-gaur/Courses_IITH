dt=0.01;
t=0:dt:10;
y=zeros(length(t),1);
y(1)=1;

for i=1:1:length(t)-1
    
   k1=dummy(t(i),y(i));
   k2=dummy(t(i)+0.5*dt,y(i)+dt*k1/2);
   k3=dummy(t(i)+0.5*dt,y(i)+dt*k2/2);
   k4=dummy(t(i)+dt,y(i)+dt*k3);
   y(i+1)=y(i)+(dt/6)*(k1+2*k2+2*k3+k4);
    
end
