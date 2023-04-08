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
            x_dot = obj.dynamics(u);
            x_next = [obj.x; obj.y] + dt*x_dot;
            time_next = obj.time + dt;
            next_state = State(x_next(1,:), x_next(2,:), time_next);
        end

        function [state_derivative] = dynamics(obj,u)
        %DYNAMICS - Dynamics of given linear system.
        %Inputs: 
        % u - current input
        %Returns state derivatives
            A = [0, 1; -1.6, -0.4];
            B = [0; 1];
            state_derivative = A*obj.to_double() + B*u;
        end

        function state_vector = to_double(obj)
        %TO_DOUBLE: Turns object of class State into standard MATLAB format
            state_vector = [obj.x;obj.y];
        end
    end
end