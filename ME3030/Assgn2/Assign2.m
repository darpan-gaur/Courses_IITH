% Name          :- Darpan Gaur
% Roll Number   :- CO21BTECH11004

dt = 0.001;
t = 0:dt:2.0;

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
x1_1(1) = 15.7893787983025;
x2_1(1) = -15.0393787983004;
y1_1(1) = -0.323710207698436;
y2_1(1) = 1.57371020769681;

% constants
l = 0.5;
m1 = 1.0;
m2 = 1.0;
c = 0.0;
k = 1000.0;
g = 0.0;

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

variables = [x1(1); x2(1); y1(1); y2(1); x1_1(1); x2_1(1); y1_1(1); y2_1(1)];

for i=1:length(t)-1
    % b = |r2-r1|

    k1 = dt * find_f(t(i), variables, m1, m2, k, c, l, g);
    k2 = dt * find_f(t(i) + dt/2, variables + k1/2, m1, m2, k, c, l, g);
    k3 = dt * find_f(t(i) + dt/2, variables + k2/2, m1, m2, k, c, l, g);
    k4 = dt * find_f(t(i) + dt, variables + k3, m1, m2, k, c, l, g);
    
    variables = variables + (k1 + 2*k2 + 2*k3 + k4) / 6;
    
    x1(i+1) = variables(1);
    x2(i+1) = variables(2);
    y1(i+1) = variables(3);
    y2(i+1) = variables(4);
    x1_1(i+1) = variables(5);
    x2_1(i+1) = variables(6);
    y1_1(i+1) = variables(7);
    y2_1(i+1) = variables(8);

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

function f = find_f(~, variables, m1, m2, k, c, l, g)
    x1   = variables(1);    x2   = variables(2);
    y1   = variables(3);    y2   = variables(4);
    x1_1 = variables(5);    x2_1 = variables(6);
    y1_1 = variables(7);    y2_1 = variables(8);
    
    b = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    fSpring = k * (b - l);
    fD_x = c * (x2_1 - x1_1);
    fD_y = c * (y2_1 - y1_1);
    
    x1_2 =  (fSpring*(x2 - x1))/(m1*b) + fD_x / m1;
    x2_2 = -(fSpring*(x2 - x1))/(m2*b) - fD_x / m2;
    y1_2 = -g + (fSpring * (y2 - y1)) / (m1 * b) + fD_y / m1;
    y2_2 = -g - (fSpring * (y2 - y1)) / (m2 * b) - fD_y / m2;
    
    f = [x1_1; x2_1; y1_1; y2_1; x1_2; x2_2; y1_2; y2_2];
end