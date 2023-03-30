function [xdot, ydot, thetadot] = dynamics(u1,u2,theta)
%DYNAMICS - Dynamics of a simple two-wheeler.
%Inputs: u1 - current lunear velocity, u2 - current angular velocity,
% theta - current heading
%Returns linear velocities in x and y and angular velocity
    xdot = cos(theta)*u1;
    ydot = sin(theta)*u1;
    thetadot = u2;
end
