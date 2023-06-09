function p = plot_against_time(state_trajectory, input_trajectory, fig_idx)
%PLOT_AGAINST_TIME plots the states x/y/theta and inputs U against time.
    if nargin < 3
        fig_idx = 100;
    end
    p = figure(fig_idx); hold on;
    N = size(state_trajectory,2);
    
    plotting = zeros(6,N);

    for idx = 1:N
        plotting(:,idx) = [state_trajectory{idx}.x;...
                           state_trajectory{idx}.y;...
                           state_trajectory{idx}.theta;...
                           state_trajectory{idx}.time;...
                           input_trajectory(1,idx);...
                           input_trajectory(2,idx)];
    end

    subplot(3,1,1);
    hold on
    plot(plotting(4,:),plotting(1,:),'k')
    plot(plotting(4,:),plotting(2,:),'r')
    xlim([0 2*pi])
    legend('x','y','Location','northwest','Interpreter','latex')
    xlabel('time [s]','Interpreter','latex')
    hold off
    
    subplot(3,1,2);
    hold on
    plot(plotting(4,:),plotting(3,:),'k')
    xlim([0 2*pi])
    ylabel('Heading $\theta$ [Rad]','Interpreter','latex')
    xlabel('Time t[s]','Interpreter','latex')
    hold off
    
    subplot(3,1,3);
    hold on
    stairs(1:N,input_trajectory(1,:),'k');
    stairs(1:N,input_trajectory(2,:),'r');
    legend('$U_1$','$U_2$','Location','north','Interpreter','latex')
    xlabel('time steps ($\Delta t = 0.01$)','Interpreter','latex')
    xlim([0 N])
    ylim([-6 6])
    hold off

end