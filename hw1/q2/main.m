%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Active Learning in Robotics Spring 2023
% Klemens Iten (KlemensIten2023@u.northwestern.edu)
%
% Homework 1, Question 2
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% reset everything and configure path
setup
% fix random number generator
rng(1);

%% Global Variables
T   = 10; % seconds
dt  = T/1000; % seconds
N   = ceil(T/dt)+1; % number of timesteps simulated
x_0 = State(10,0); % initial condition

% Desired State Trajectory (is just a nullvector for Q2)
desired_state_vector = cell(1,N);
for idx = 1 : N
    desired_state_vector{idx} = State(0, 0, (idx-1)/(N-1)*T);
end

% State Trajectory Preallocation
state_vector = cell(1,N);
state_vector{1} = x_0;
for idx = 2 : N
    state_vector{idx} = State();
end

%% Optimize Trajectory
[optimized_state_trajectory, optimized_input] = optimize_trajectory(...
                                            x_0,desired_state_vector,T,dt);
plot_state_trajectory(optimized_state_trajectory,3);
plot_against_time(optimized_state_trajectory,optimized_input,4);

save('q2.mat',"optimized_state_trajectory","optimized_input")

% %% Plot Post-Processing
% plot_all(desired_state_vector,...
%          state_vector, input_vector,...
%          optimized_state_trajectory, optimized_input,5);

%% Cleanup
cleanup
