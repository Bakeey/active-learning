function p = plot_state_trajectory(state_trajectory, fig_idx)
%PLOT_STATE_TRAJECTORY plots the trajectory of a State
    if nargin < 2
        fig_idx = 99;
    end
    p = figure(fig_idx); hold on;
    N = size(state_trajectory,2);
    
    plotting = zeros(4,N);
    for idx = 1:N
        plotting(:,idx) = [state_trajectory{idx}.x;...
                           state_trajectory{idx}.y;...
                           state_trajectory{idx}.theta;...
                           state_trajectory{idx}.time];
    end
    
    subplot(2,1,1);
    plot(plotting(1,:),plotting(2,:));
    ylabel('$y$','Interpreter','latex');
    xlabel('$x$','Interpreter','latex');

    subplot(2,1,2); 
    plot(plotting(4,:),plotting(3,:));
    ylabel('Heading $\theta$ [Rad]','Interpreter','latex')
    xlabel('Time t[s]','Interpreter','latex')

    hold off;

end