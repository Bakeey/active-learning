function optimal_trajectory = optimizer(x_0,x_des, N)
%OPTIMIZER
% An optimal control problem (OCP),
% solved with direct multiple-shooting.
%
% Adapted from: http://labs.casadi.org/OCP
    opti = casadi.Opti(); % Optimization problem

    % ---- decision variables ---------
    X = opti.variable(3,N); % state trajectory
    x = X(1,:);
    y = X(2,:);
    theta = X(3,:);
    U = opti.variable(2,N);   % control trajectory

    % ---- objective ---------
    objective = 0;
    for idx = 1 : N
        objective = objective + X(:,idx)'*X(:,idx)'; % TODO
    end
    opti.minimize(objective);

    % ---- dynamic constraints --------
    x_0 = current_state;
    for k=1:N % loop over control intervals
       next_state = current_state.next(U(1,k),U(2,k))
       x_next = TODO STATE to MX
       opti.subject_to(X(:,k+1)==x_next); % close the gaps
    end


end

