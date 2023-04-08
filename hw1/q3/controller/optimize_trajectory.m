function [state_trajectory,input_trajectory] = ...
                      optimize_trajectory(x_0, desired_state_vector, T, dt)
%OPTIMIZER
% An optimal control problem (OCP),
% solved with direct multiple-shooting for the lecture
% Active Learning in Robotics Spring 2023 at Northwestern University by
% Klemens Iten (KlemensIten2023@u.northwestern.edu)
%
% Adapted from: https://web.casadi.org/docs/#document-ocp 
% and https://web.casadi.org/blog/ocp/

% This code uses CasADi - A software framework for nonlinear optimization.
% For download here: https://web.casadi.org/get/

    if nargin < 3 % fast debugging
        T  = 10;                      % sec
        dt = T/1000;                      % sec
        if nargin < 2
            % Desired State Trajectory
            N    = ceil(T/dt)+1;
            desired_state_vector = cell(1,N);
            for idx = 1 : N
                desired_state_vector{idx} = State(0, 0, (idx-1)/(N-1)*T);
            end
            if nargin < 1
                x_0 = State(10,0);    % initial state
            end
        end
    end

    % ---- function-wide gobals --------
    N    = ceil(T/dt)+1;
    Q    = dt*diag([2,0.01]);
    R    = dt*diag(0.1);
    M    = diag([1,0.01]);
    opti = casadi.Opti(); % Optimization problem

    % ---- decision variables ---------
    X_des = zeros(2, N);
    for idx = 1:N
        X_des(:,idx) = desired_state_vector{idx}.to_double();
    end

    X = opti.variable(2,N); % state trajectory
    x = X(1,:);
    y = X(2,:);
    U = opti.variable(1,N);   % control trajectory

    % ---- objective ---------
    objective = 0;
    
    % stage costs l(x_idx,u_idx)
    for idx = 1 : N-1
        objective = objective + (X(:,idx)-X_des(:,idx))'*Q*(X(:,idx)-X_des(:,idx));
        objective = objective + (U(:,idx))'*R*(U(:,idx));
    end
  
    % terminal cost m(x_N)
    objective = 0.5*objective*dt + 0.5*(X(:,N)-X_des(:,N))'*M*(X(:,N)-X_des(:,N));

    opti.minimize(objective);

    % ---- dynamic constraints --------
    % dX/dt = f(theta,U)

    for k=1 : N-1 % loop over control intervals
       x_next = X(:,k) + dt * dynamics(X(:,k), U(:,k));
       opti.subject_to(X(:,k+1) - x_next == 0); % close the gaps
    end

    % ---- path constraints -----------
    % opti.subject_to(-5<=U<=5);           % limits control arbitrarily
    % opti.subject_to(-5<=U(1,:)<=5);

    % ---- boundary and initial conditions --------
    opti.subject_to(X(:,1)    == x_0.to_double());

    % ---- solve NLP              ------
    opti.solver('ipopt'); % set numerical backend
    sol = opti.solve();   % actual solve

    % ---- construct solution ------
    state_trajectory = cell(1,N);
    input_trajectory = sol.value(U);
    for idx = 1 : N
        state_trajectory{idx} = State(sol.value(x(idx)),...
                                        sol.value(y(idx)),...
                                        (idx-1)*dt);
    end

    % ---- plots for debugging ------
    if nargin == 0
        close all
        plot_state_trajectory(state_trajectory,99);
        plot_against_time(state_trajectory,input_trajectory,100);
    end
end