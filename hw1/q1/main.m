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
dt  = 0.05; % seconds
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
input_vector = [u1*ones([1,N]);u2*ones([1,N])];

%  simulate for N timesteps
for idx = 2 : N
    state_vector{idx} = state_vector{idx-1}.next(u1,u2,dt);
end

%  plot resulting trajectory
plot_state_trajectory(state_vector,1);
plot_against_time(state_vector,input_vector,2);

%% Optimize Trajectory
[optimized_state_trajectory, optimized_input] = optimize_trajectory(...
                                                             x_0,x_d,T,dt);
plot_state_trajectory(optimized_state_trajectory,3);
plot_against_time(optimized_state_trajectory,optimized_input,4);

%% Cleanup
cleanup