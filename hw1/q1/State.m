classdef State
    properties
        x {mustBeNumeric}
        y {mustBeNumeric}
        theta {mustBeNumeric}
        time {mustBeNumeric}
    end
    methods
        function obj = State(x0,y0,theta0,time)
            if nargin == 4
                obj.x = x0;
                obj.y = y0;
                obj.theta = theta0;
                obj.time = time;
            elseif nargin == 3
                obj.x = x0;
                obj.y = y0;
                obj.theta = theta0;
                obj.time = 0;
            else
                obj.x = NaN;
                obj.y = NaN;
                obj.theta = NaN;
            end

        end
        function next_state = next(obj,u1,u2,dt)
            if nargin == 3
                dt = 1E-2;
            end
            [xdot, ydot, thetadot] = obj.dynamics(u1,u2);
            x_next = obj.x + dt*xdot;
            y_next = obj.y + dt*ydot;
            theta_next = obj.theta + dt*thetadot;
            time_next = obj.time + dt;
            next_state = State(x_next,y_next,theta_next,time_next);
        end
        function [xdot, ydot, thetadot] = dynamics(obj,u1,u2)
        %DYNAMICS - Dynamics of a simple two-wheeler.
        %Inputs: u1 - current lunear velocity, u2 - current angular velocity,
        % theta - current heading
        %Returns linear velocities in x and y and angular velocity
            xdot = cos(obj.theta)*u1;
            ydot = sin(obj.theta)*u1;
            thetadot = u2;
        end
    end
end
