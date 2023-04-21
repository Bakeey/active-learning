% Compute the value of the integrand at 2*pi/3.
dt = 0.001;
T = 2*pi;
N = ceil(T/dt);
% x_0 = [0;0;pi/2];
x_0 = [0.68773393; -0.31485843; -0.56749667];
U = [1;-0.5];

x = zeros(3,N);
x(:,1) = x_0;

for idx = 2:N
    theta = x(3,idx-1);
    x_dot = A(theta, U)*x(:,idx-1) + B(theta, U)*U;
    x(:,idx) = x(:,idx-1) + dt * x_dot;
end

plot(x(1,:),x(2,:))



function y = A(theta, U)
    y = [0, 0, -sin(theta)*U(1); 0, 0, cos(theta)*U(1); 0, 0, 0];
end

function y = B(theta, ~)
    y = [cos(theta), 0; sin(theta), 0; 0, 1];
end