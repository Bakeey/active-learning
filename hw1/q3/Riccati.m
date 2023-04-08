classdef Riccati
    properties
        Phi
        time
    end
    methods
        function obj = Riccati(Phi,time)
            if nargin == 2
                obj.Phi = Phi;
                obj.time = time;
            else
                obj.Phi = NaN;
                obj.time = NaN;
            end

        end
        function previous_state = previous(obj,dt)
            if nargin < 2
                dt = -1E-2; % TODO fix this cluterfuck
            end
            Phi_dot = obj.dynamics();
            phi_prev = obj.Phi + dt*Phi_dot;
            time_prev = obj.time + dt;
            previous_state = Riccati(phi_prev, time_prev);
        end

        function [state_derivative] = dynamics(obj)
        %DYNAMICS - Dynamics of given system.
        %Returns state derivative
            A = [0, 1; -1.6, -0.4];
            B = [0; 1];
            Q    = diag([2,0.01]);
            R    = diag(0.1);

            state_derivative = obj.Phi*B*inv(R)*B'*obj.Phi - obj.Phi*A - A'*obj.Phi-Q;
        end

        function state_vector = to_double(obj)
        %TO_DOUBLE: Turns object of class State into standard MATLAB format
            state_vector = [obj.x;obj.y];
        end
    end
end