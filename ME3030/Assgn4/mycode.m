% Name          :- Darpan Gaur
% Roll Number   :- CO21BTECH11004

% Constants
l = 0.5;
m1 = 1.0;
m2 = 1.0;
c = 0.0;
k = 1000.0;
g = 0.0;

Ti = 0.0;
Tf = 2.0;

% Given boundary conditions
inPos = [0.0; 0.0; 0.5; 0.0];
fPos = [1.0; 1.0; 1.0; 1.5];

% Guess for velocity
v = [5.0; 5.0; -5.0; 5.0];

% Small change
dv = 1.0e-3;

% Tolerance
eps = 1e-3;
eps2 = 1e-6;

while true
    
    pos = implicit_euler_solve(inPos, v, Ti, Tf, m1, m2, k, c, l, g, eps2);
    
    err = pos - fPos;

    if (max(abs(err))) < eps
        fprintf("Final Position: [x1 y1 x2 y2]'\n");
        disp(pos);
        break
    end

    J = zeros(4, 4);
    
    for i = 1:4
        v_new = v;
        v_new(i) = v_new(i) + dv;
        pos_dv = implicit_euler_solve(inPos, v_new, Ti, Tf, m1, m2, k, c, l, g, eps2);
        J_col = zeros(4, 1);
        for j = 1:4
            derivative = (pos_dv(j) - pos(j)) / dv;
            J_col(j) = derivative;
        end
        J(:, i) = J_col;
    end
    
    v = v - J \ err;

end

format longg

fprintf("Final velocity: [vx1 vy1 vx2 vy2]'\n");
disp(v);

function fPos = implicit_euler_solve(inPos, init_velocity, init_time, Tf, m1, m2, k, c, l, g, eps)

    dt = 1.0e-4;    % Time step (you may need to adjust this)

    % Small change in values for calculating numerical derivative
    small_change = 1.0e-4;
    num_steps = round((Tf - init_time) / dt + 1);
    
    
    % Initialize arrays to store positions and velocities
    x1 = zeros(1, num_steps);
    y1 = zeros(1, num_steps);
    x1_1 = zeros(1, num_steps);
    y1_1 = zeros(1, num_steps);
    
    x2 = zeros(1, num_steps);
    y2 = zeros(1, num_steps);
    x2_1 = zeros(1, num_steps);
    y2_1 = zeros(1, num_steps);
    
    % Set initial conditions
    x1(1) = inPos(1);
    y1(1) = inPos(2);
    x1_1(1) = init_velocity(1);
    y1_1(1) = init_velocity(2);
    
    x2(1) = inPos(3);
    y2(1) = inPos(4);
    x2_1(1) = init_velocity(3);
    y2_1(1) = init_velocity(4);
    
    variables = [x1(1) ; y1(1) ; x1_1(1) ; y1_1(1) ; x2(1) ; y2(1) ; x2_1(1) ; y2_1(1)];
    
    % Using implicit euler method
    for i = 1:num_steps-1
        
        guess = variables;
        f_values = find_f(guess, variables, dt, m1, m2, k, c, l, g);

        while (max(abs(f_values)) > eps)
            J = zeros(8, 8);
            
            for j = 1:8
                temp = guess;
                temp(j) = temp(j) + small_change;
                temp_f_values = find_f(temp, variables, dt, m1, m2, k, c, l, g);
                J_col = zeros(8, 1);
                for ind = 1:8
                    derivative = (temp_f_values(ind) - f_values(ind)) / small_change;
                    J_col(ind) = derivative;
                end
                J(:, j) = J_col;
            end

            guess = guess - J \ f_values;

            f_values = find_f(guess, variables, dt, m1, m2, k, c, l, g);

        end
        variables = guess;

        x1(i+1) = variables(1);     y1(i+1) = variables(2);
        x1_1(i+1) = variables(3);   y1_1(i+1) = variables(4);
        x2(i+1) = variables(5);     y2(i+1) = variables(6);
        x2_1(i+1) = variables(7);   y2_1(i+1) = variables(8);
    end
    fPos = [variables(1); variables(2); variables(5); variables(6)];

end

function values = find_f(variables, prev_arr, dt, m1, m2, k, c, l, g)
    x1 = variables(1);      y1 = variables(2);
    x1_1 = variables(3);    y1_1 = variables(4);
    x2 = variables(5);      y2 = variables(6);
    x2_1 = variables(7);    y2_1 = variables(8);

    b = sqrt((x1 - x2)^2 + (y1 - y2)^2);
    fSprinf = k * (b - l);
    fD_x = c * (x2_1 - x1_1);
    fD_y = c * (y2_1 - y1_1);

    x1_2 = (fSprinf * (x2 - x1)) / (m1 * b) + fD_x / m1;
    y1_2 = (fSprinf * (y2 - y1)) / (m1 * b) + fD_y / m1 - g;

    x2_2 = -(fSprinf * (x2 - x1)) / (m2 * b) - fD_x / m2;
    y2_2 = -(fSprinf * (y2 - y1)) / (m2 * b) - fD_y / m2 - g;

    temp = [x1_1 ; y1_1 ; x1_2 ; y1_2 ; x2_1 ; y2_1 ; x2_2 ; y2_2];
    values = variables - prev_arr - dt * temp;
end
