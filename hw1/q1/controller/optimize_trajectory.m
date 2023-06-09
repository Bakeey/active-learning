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
        T  = 2*pi;                      % sec
        dt = 0.025;                      % sec
        if nargin < 2
            % Desired State Trajectory
            N    = ceil(T/dt);
            desired_state_vector = cell(1,N);
            for idx = 1 : N
                desired_state_vector{idx} = ...
                          State(4*(idx-1)/(N-1), 0, pi/2, (idx-1)/(N-1)*T);
            end
            if nargin < 1
                x_0 = State(0,0,pi/2);    % initial state
            end
        end
    end

    % ---- function-wide gobals --------
    N    = ceil(T/dt);
    Q    = dt*diag([1,3,4]);
    R    = dt*diag([0.01,0.01]);
    M    = diag([1,3,4]);
    opti = casadi.Opti(); % Optimization problem

    % ---- decision variables ---------
    X_des = zeros(3, N);
    for idx = 1:N
        X_des(:,idx) = desired_state_vector{idx}.to_double();
    end

    X = opti.variable(3,N); % state trajectory
    x = X(1,:);
    y = X(2,:);
    theta = X(3,:);
    U = opti.variable(2,N);   % control trajectory

    % ---- objective ---------
    objective = 0;
    
    % stage costs l(x_idx,u_idx)
    for idx = 1 : N-1
        objective = objective + (X(:,idx)-X_des(:,idx))'*Q*(X(:,idx)-X_des(:,idx));
        objective = objective + (U(:,idx))'*R*(U(:,idx));
    end
  
    % terminal cost m(x_N)
    objective = objective + (X(:,N)-X_des(:,N))'*M*(X(:,N)-X_des(:,N));

    opti.minimize(objective);

    % ---- dynamic constraints --------
    % dX/dt = f(theta,U)

    for k=1 : N-1 % loop over control intervals
       x_next = X(:,k) + dt * dynamics(theta(k),U(:,k));
       opti.subject_to(X(:,k+1) - x_next == 0); % close the gaps
    end

    % ---- path constraints -----------
    % opti.subject_to(-5<=U<=5);           % limits control arbitrarily
    % opti.subject_to(-5<=U(1,:)<=5);

    % ---- boundary and initial conditions --------
    opti.subject_to(X(:,1)    == x_0.to_double());
    opti.subject_to(X(:,N)    == desired_state_vector{N}.to_double());

    % ---- solve NLP              ------
    opti.solver('ipopt'); % set numerical backend
    sol = opti.solve();   % actual solve

    % ---- construct solution ------
    state_trajectory = cell(1,N);
    input_trajectory = sol.value(U);
    for idx = 1 : N
        state_trajectory{idx} = State(sol.value(x(idx)),...
                                        sol.value(y(idx)),...
                                        sol.value(theta(idx)),...
                                        (idx-1)*dt);
    end

    % ---- plots for debugging ------
    if nargin == 0
        close all
        plot_state_trajectory(state_trajectory,99);
        plot_against_time(state_trajectory,input_trajectory,100);
    end
end