%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Active Learning in Robotics Spring 2023
% Klemens Iten (KlemensIten2023@u.northwestern.edu)
%
% Homework 1, Question 3
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
Phi_end = diag([1,0.01]); % final condition
x_0 = State(10,0);

% Riccatti Equation Trajectory Preallocation
riccati_vector = cell(1,N);
riccati_vector{N} = Riccati(Phi_end,T);
for idx = (N-1) : -1 : 1
    disp(idx)
    riccati_vector{idx} = riccati_vector{idx+1}.previous(-dt);
end

A = [0, 1; -1.6, -0.4];
B = [0; 1];
Q    = diag([2,0.01]);
R    = diag(0.1);

% Desired State Trajectory (is just a nullvector for Q3)
state_vector = cell(1,N);
state_vector{1} = x_0;

u = zeros(1,N);

for idx = 2 : N
    K = inv(R) * B' * riccati_vector{idx-1}.Phi;
    u(idx-1) = -K * [state_vector{idx-1}.x; state_vector{idx-1}.y];
    state_vector{idx} = state_vector{idx-1}.next(u(idx-1),dt);
end


plot_state_trajectory(state_vector,3);
plot_against_time(state_vector,u,4);


%% Plot difference
load("q2.mat");

difference_state = cell(1,N);
difference_input = - u + optimized_input;

for idx = 1:N
    difference_state{idx} = State(...
        - state_vector{idx}.x + optimized_state_trajectory{idx}.x,...
        - state_vector{idx}.y + optimized_state_trajectory{idx}.y,...
        state_vector{idx}.time);
end
        
    
plot_state_trajectory(difference_state,5);
plot_against_time(difference_state,difference_input,6);


% %% Plot Post-Processing
% plot_all(desired_state_vector,...
%          state_vector, input_vector,...
%          optimized_state_trajectory, optimized_input,5);

%% Cleanup
cleanup
