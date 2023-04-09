classdef State
    properties
        x {mustBeNumeric}
        y {mustBeNumeric}
        time {mustBeNumeric}
    end
    methods
        function obj = State(x0,y0,time)
            if nargin == 3
                obj.x = x0;
                obj.y = y0;
                obj.time = time;
            elseif nargin == 2
                obj.x = x0;
                obj.y = y0;
                obj.time = 0;
            else
                obj.x = NaN;
                obj.y = NaN;
                obj.time = NaN;
            end

        end
        function next_state = next(obj,u,dt)
            if nargin < 2
                dt = 1E-2;
            end
               
            Y_1 = obj.to_double();
            Y_2 = obj.to_double() + dt/2 * obj.dynamics(u,Y_1);
            Y_3 = obj.to_double() + dt/2 * obj.dynamics(u,Y_2);
            Y_4 = obj.to_double() + dt * obj.dynamics(u,Y_3);

            x_dot = (obj.dynamics(u, Y_1) + 2*obj.dynamics(u, Y_2) + ...
                2*obj.dynamics(u, Y_3) + obj.dynamics(u, Y_4))/6; % RK-4

            % x_dot = obj.dynamics(u, obj.to_double()); % Euler
            x_next = [obj.x; obj.y] + dt*x_dot;
            time_next = obj.time + dt;
            next_state = State(x_next(1,:), x_next(2,:), time_next);
        end

        function [state_derivative] = dynamics(obj,u, state)
        %DYNAMICS - Dynamics of given linear system.
        %Inputs: 
        % u - current input
        %Returns state derivatives
            if nargin < 2
                state = obj.to_double();
            end
            A = [0, 1; -1.6, -0.4];
            B = [0; 1];
            state_derivative = A*state + B*u;
        end

        function state_vector = to_double(obj)
        %TO_DOUBLE: Turns object of class State into standard MATLAB format
            state_vector = [obj.x;obj.y];
        end
    end
end