function [state_trajectory,input_trajectory] = ...
                                     optimize_trajectory(x_0, x_des, T, dt)
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
        dt = 1E-1;                      % sec
        if nargin < 2
            x_des = State(4,0,pi/2);       % desired state
            if nargin < 1
                x_0 = State(0,0,pi/2);    % initial state
            end
        end
    end

    % ---- function-wide gobals --------
    N    = ceil(T/dt);
    Q    = dt*diag([4,2,2]);
    R    = dt*diag([40,20]);
    M    = diag([40,40,400]);
    opti = casadi.Opti(); % Optimization problem

    % ---- decision variables ---------
    X_des = x_des.to_double();

    X = opti.variable(3,N); % state trajectory
    x = X(1,:);
    y = X(2,:);
    theta = X(3,:);
    U = opti.variable(2,N);   % control trajectory

    % ---- objective ---------
    objective = 0;
    
    % stage costs l(x_idx,u_idx)
    for idx = 1 : N-1
        objective = objective + (X(:,idx)-X_des)'*Q*(X(:,idx)-X_des);
        objective = objective + (U(:,idx))'*R*(U(:,idx));
    end
  
    % terminal cost m(x_N)
    objective = objective + (X(:,N)-X_des)'*M*(X(:,N)-X_des);

    opti.minimize(objective);

    % ---- dynamic constraints --------
    % dX/dt = f(theta,U)

    for k=1 : N-1 % loop over control intervals
       x_next = X(:,k) + dt * dynamics(theta(k),U(:,k));
       opti.subject_to(X(:,k+1) - x_next == 0); % close the gaps
    end

    % ---- path constraints -----------
    opti.subject_to(-5<=U<=5);           % limits control arbitrarily

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