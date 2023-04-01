function f = dynamics(theta,u)
%DYNAMICS - Dynamics of a simple two-wheeler.
%Inputs: 
% u(1) - current linear velocity, 
% u(2) - current angular velocity,
% theta - current heading
%Returns linear velocities in x and y and angular velocity
f = [cos(theta)*u(1);    % dX/dt = f(theta,U)
     sin(theta)*u(1);
     u(2)          ];
end