% Name          :- Darpan Gaur
% Roll Number   :- CO21BTECH11004

% constants
l = 0.5;
m1 = 1.0;
m2 = 1.0;
c = 0.0;
k = 1000.0;
g = 9.81;  
Ti = 0.0;
Tf = 2.0;

% Given boundary conditions
inPos = [0.0; 0.0; 0.5; 0.0];
fPos = [1.0; 1.0; 1.0; 1.5];

% Guess for velocity
v = [10.0; 10.0; -10.0; 5.0];

% Small change
dv = 1.0e-3;


% tolerance
eps = 1.0e-3;


while true  
   pos = rk4_solve(inPos, v, Ti, Tf, m1, m2, k, c, l, g);
   err = pos - fPos;
   % L1 scheme for error
   if (max(abs(err))) < eps
       fprintf("Final Position :- [x1 y1 x2 y2]' \n");
       disp(pos)
       break
   end
   J = zeros(4, 4);
  
   for i = 1:4
       v_new = v;
       v_new(i) = v_new(i) + dv;
       pos_dv = rk4_solve(inPos, v_new, Ti, Tf, m1, m2, k, c, l, g);
       J_col = zeros(4, 1);
       for j = 1:4
           derivative = (pos_dv(j) - pos(j)) / dv;
           J_col(j) = derivative;
       end
       J(:, i) = J_col;
   end
  
   v = v - J \ err;
end

fprintf("Final velocity :- [vx1 vy1 vx2 vy2]' \n")
disp(v)

function finalPos = rk4_solve(inPos, init_velocity, Ti, Tf, m1, m2, k, c, l, g)
  
   dt = 0.00001;
  
   % Number of time steps
   num_steps = round((Tf - Ti) / dt + 1);
  
   % Initialize arrays to store positions and velocities
   x1 = zeros(1, num_steps);
   y1 = zeros(1, num_steps);
   x1_1 = zeros(1, num_steps);
   y1_1 = zeros(1, num_steps);
  
   x2 = zeros(1, num_steps);
   y2 = zeros(1, num_steps);
   x1_2 = zeros(1, num_steps);
   y1_2 = zeros(1, num_steps);
  
   % Set initial conditions
   x1(1) = inPos(1);
   y1(1) = inPos(2);
   x1_1(1) = init_velocity(1);
   y1_1(1) = init_velocity(2);
  
   x2(1) = inPos(3);
   y2(1) = inPos(4);
   x1_2(1) = init_velocity(3);
   y1_2(1) = init_velocity(4);
  
   variables = [x1(1) ; y1(1) ; x1_1(1) ; y1_1(1) ; x2(1) ; y2(1) ; x1_2(1) ; y1_2(1)];
  
   % Using RK4 method
   for i = 1:num_steps-1
       k1 = dt * find_f(variables, m1, m2, k, c, l, g);
       k2 = dt * find_f(variables + 0.5 * k1, m1, m2, k, c, l, g);
       k3 = dt * find_f(variables + 0.5 * k2, m1, m2, k, c, l, g);
       k4 = dt * find_f(variables + k3, m1, m2, k, c, l, g);
       variables = variables + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
       x1(i+1) = variables(1);   y1(i+1) = variables(2);
       x1_1(i+1) = variables(3); y1_1(i+1) = variables(4);
       x2(i+1) = variables(5);   y2(i+1) = variables(6);
       x1_2(i+1) = variables(7); y1_2(i+1) = variables(8);
   end
   finalPos = [variables(1); variables(2); variables(5); variables(6)];
end

function f = find_f(variables, m1, m2, k, c, l, g)
   x1 = variables(1);    y1 = variables(2);
   x1_1 = variables(3);   y1_1 = variables(4);
   x2 = variables(5);    y2 = variables(6);
   x2_1 = variables(7);   y2_1 = variables(8);
   b = sqrt((x1 - x2)^2 + (y1 - y2)^2);
   fSpring = k * (b - l);
   fD_x = c * (x2_1 - x1_1);
   fD_y = c * (y2_1 - y1_1);
   x1_2 = (fSpring * (x2 - x1)) / (m1 * b) + fD_x / m1;
   y1_2 = (fSpring * (y2 - y1)) / (m1 * b) + fD_y / m1 - g;
   x2_2 = -(fSpring * (x2 - x1)) / (m2 * b) + fD_x / m2;
   y2_2 = -(fSpring * (y2 - y1)) / (m2 * b) + fD_y / m2 - g;
  
   f = [x1_1 ; y1_1 ; x1_2 ; y1_2 ; x2_1 ; y2_1 ; x2_2 ; y2_2];
end

