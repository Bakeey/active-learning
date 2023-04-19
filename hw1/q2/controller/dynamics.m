function [state_derivative] = dynamics(X,u)
    %DYNAMICS - Dynamics of given linear system.
    %Inputs: 
    % u - current input
    %Returns state derivatives
        A = [0, 1; -1.6, -0.4];
        B = [0; 1];
        state_derivative = A*X + B*u;
    end
