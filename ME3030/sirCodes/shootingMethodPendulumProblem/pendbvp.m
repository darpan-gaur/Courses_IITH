function zdot=pendbvp(t,z)
global alpha
theta=z(1);
thetadot=z(2);
zdot=[thetadot; -alpha*sin(theta)];