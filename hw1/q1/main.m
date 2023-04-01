%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Active Learning in Robotics Spring 2023
% Klemens Iten (KlemensIten2023@u.northwestern.edu)
%
% Homework 1, Question 1
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% reset everything and configure path
setup
% fix random number generator
rng(1);

%% Global Variables
T   = 2*pi; % seconds
dt  = 1E-2; % seconds
N   = ceil(T/dt); % number of timesteps simulated
x_0 = State(0,0,pi/2); % initial condition
x_d = State(4,0,pi/2); % desired end state

state_vector = cell(1,N);
state_vector{1} = x_0;
for idx = 2 : N
    state_vector{idx} = State();
end

%% Initial Trajectory (semi-circle)
%  given constant control input
u1  = 1;
u2  = -.5;

%  simulate for N timesteps
for idx = 2 : N
    state_vector{idx} = state_vector{idx-1}.next(u1,u2);
end

%  plot resulting trajectory
plot_state_trajectory(state_vector,1);


