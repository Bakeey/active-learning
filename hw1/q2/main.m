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

%% Directional Derivative
Q    = diag([2,0.01]);
R    = diag(0.1);
M    = diag([1,0.01]);

D_cost_pertubed = zeros(10,5);

z = zeros([2,N]); % state pertubation
directions = @(A,B,C,D) A*sin(B*linspace(0,T,N)+C)+D; % input pertubation

for episode = 1:10 
    A_v = 10/episode;
    B_v = episode/10;
    C_v = 2*pi*(episode-1)/9;
    D_v = episode;
    D_cost_pertubed(episode,1:4) = [A_v,B_v,C_v,D_v];

    v = directions(A10_v,B_v,C_v,D_v);
    z_state = cell(1,N);
    z_state{1} = State(0,0);
    
    for idx = 2:N
        z_state{idx} = z_state{idx-1}.next(v(idx-1),dt);
        z(:,idx) = z_state{idx}.to_double();
    end
    
    for idx = 1:(N-1)
        x_idx = optimized_state_trajectory{idx}.to_double();
        u_idx = optimized_input(idx);
        D_cost_pertubed(episode,5) = D_cost_pertubed(episode,5) + dt * ((Q*x_idx)'*z(:,idx) + ...
                          R*u_idx'*v(:,idx));
    end
    x_N = optimized_state_trajectory{N}.to_double();
    D_cost_pertubed(episode,5) = D_cost_pertubed(episode,5) + (M*x_N)'*z(:,N);
    
    plot_against_time(z_state,v,4+episode);

end

%print the header
disp(D_cost_pertubed)

%% Cleanup
% cleanup
