function ydot=pendval(t,y)
global alpha omega A

ydot=[y(2); -alpha*sin(y(1))+A*sin(omega*t)];