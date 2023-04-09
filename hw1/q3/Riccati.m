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
            
            Y_1 = obj.Phi;
            Y_2 = obj.Phi + dt/2 * obj.dynamics(Y_1);
            Y_3 = obj.Phi + dt/2 * obj.dynamics(Y_2);
            Y_4 = obj.Phi + dt * obj.dynamics(Y_3);

            %Phi_dot = (obj.dynamics(Y_1) + 2*obj.dynamics(Y_2) + ...
            %    2*obj.dynamics(Y_3) + obj.dynamics(Y_4))/6; % RK-4
            Phi_dot = obj.dynamics(obj.Phi); % Euler

            phi_prev = obj.Phi + dt*Phi_dot;
            time_prev = obj.time + dt;
            previous_state = Riccati(phi_prev, time_prev);
        end

        function [state_derivative] = dynamics(obj, Phi0)
        %DYNAMICS - Dynamics of given system.
        %Returns state derivative
            if nargin < 1
                Phi0 = obj.Phi;
            end

            A = [0, 1; -1.6, -0.4];
            B = [0; 1];
            Q    = diag([2,0.01]);
            R    = diag(0.1);

            state_derivative = Phi0*B*inv(R)*B'*Phi0 - Phi0*A - A'*Phi0-Q;
        end

        function state_vector = to_double(obj)
        %TO_DOUBLE: Turns object of class State into standard MATLAB format
            state_vector = [obj.x;obj.y];
        end
    end
end