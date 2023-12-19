% Name          :- Darpan Gaur
% Roll Number   :- CO21BTECH11004

dt = 0.0001;
t = 0:dt:2;

% arrays to store positoin
x1 = zeros(1, length(t));
x2 = zeros(1, length(t));
y1 = zeros(1, length(t));
y2 = zeros(1, length(t));

% arrays to store velocity
x1_1 = zeros(1, length(t));
x2_1 = zeros(1, length(t));
y1_1 = zeros(1, length(t));
y2_1 = zeros(1, length(t));

% set boumdary conditions
x1(1) = 0.0;
x2(1) = 0.5;
y1(1) = 0.0;
y2(1) = 0.0;
x1_1(1) = 25.3445046045039;
x2_1(1) = -24.5945046045077;
y1_1(1) = 3.4750112072853;
y2_1(1) = 17.3959697927552;

% constants
l = 0.5;
m1 = 1.0;
m2 = 1.0;
c = 0.0;
k = 1000.0;
g = 9.81;

% array to store Energy
KE = zeros(1, length(t));
S_PE = zeros(1, length(t));
G_PE = zeros(1, length(t));
TE = zeros(1, length(t));

KE(1) = (m1*((x1_1(1))^2 + (y1_1(1))^2) + m2*((x2_1(1))^2 + (y2_1(1))^2))*0.5;
% b = |r2-r1|
b = sqrt((x2(1) - x1(1))^2 + (y2(1) - y1(1))^2);
S_PE(1) = k*(b-l)*(b-l)*0.5;
G_PE(1) = (m1*y1(1) + m2*y2(1))*g;
TE(1) = KE(1) + S_PE(1) + G_PE(1);

for i=1:length(t)-1
    % b = |r2-r1|
    b = sqrt((x2(i) - x1(i))^2 + (y2(i) - y1(i))^2);
    fSpring = k*(b-l);
    fD_x = c*(x2_1(i) - x1_1(i));
    fD_y = c*(y2_1(i) - y1_1(i));
    
    x1_2 = (fSpring*(x2(i)-x1(i)))/(m1*b) + fD_x/m1;
    x2_2 = -(fSpring*(x2(i)-x1(i)))/(m2*b) - fD_x/m2;
    y1_2 = -g + (fSpring*(y2(i)-y1(i)))/(m1*b) + fD_y/m2;
    y2_2 = -g -(fSpring*(y2(i)-y1(i)))/(m2*b) - fD_y/m2;

    x1_1(i+1) = x1_1(i) + x1_2*dt;
    x2_1(i+1) = x2_1(i) + x2_2*dt;
    y1_1(i+1) = y1_1(i) + y1_2*dt;
    y2_1(i+1) = y2_1(i) + y2_2*dt;

    x1(i+1) = x1(i) + x1_1(i+1)*dt;
    x2(i+1) = x2(i) + x2_1(i+1)*dt;
    y1(i+1) = y1(i) + y1_1(i+1)*dt;
    y2(i+1) = y2(i) + y2_1(i+1)*dt;

    b = sqrt((x2(i+1) - x1(i+1))^2 + (y2(i+1) - y1(i+1))^2);
    
    KE(i+1) = (m1*((x1_1(i+1))^2 + (y1_1(i+1))^2) + m2*((x2_1(i+1))^2 + (y2_1(i+1))^2))*0.5;
    S_PE(i+1) = k*(b-l)*(b-l)*0.5;
    G_PE(i+1) = (m1*y1(i+1) + m2*y2(i+1))*g;
    TE(i+1) = KE(i+1) + S_PE(i+1) + G_PE(i+1);
    TE(i+1) = round(TE(i+1), 3);    % for correcting round off error
end

% plot
figure;

subplot(2,2,1);
plot(t, x1,'b');
xlabel('t');
ylabel('x1');
title('x1 vs. Time ');

subplot(2,2,2);
plot(t, y1, 'b');
xlabel('t');
ylabel('y1');
title('y1 vs. Time ');

subplot(2,2,3);
plot(t, x2, 'b');
xlabel('t');
ylabel('x2');
title('x2 vs. Time ');

subplot(2,2,4);
plot(t, y2, 'b');
xlabel('t');
ylabel('y2');
title('y2 vs. Time ');

figure;
plot(t, TE, 'b');
xlabel('t');
ylabel('Total Energy');
title('Total Energy vs. Time ');
