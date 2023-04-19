function p = plot_against_time(state_trajectory, input_trajectory, fig_idx)
%PLOT_AGAINST_TIME plots the states x/y/theta and inputs U against time.
    if nargin < 3
        fig_idx = 100;
    end
    p = figure(fig_idx); hold on;
    N = size(state_trajectory,2);
    
    plotting = zeros(4,N);

    for idx = 1:N
        plotting(:,idx) = [state_trajectory{idx}.x;...
                           state_trajectory{idx}.y;...
                           state_trajectory{idx}.time;...
                           input_trajectory(idx)];
    end

    subplot(2,1,1);
    hold on
    plot(plotting(3,:),plotting(1,:),'k')
    plot(plotting(3,:),plotting(2,:),'r')
    legend('$\Delta x_1$','$\Delta x_2$','Location','northeast','Interpreter','latex')
    xlabel('time [s]','Interpreter','latex')
    hold off
    
    subplot(2,1,2);
    hold on
    stairs(1:N,input_trajectory(1,:),'k');
    legend('$\Delta u_1$','Location','northeast','Interpreter','latex')
    xlabel('time steps ($\Delta t = 0.01$)','Interpreter','latex')
    xlim([0 1000])
    hold off

end