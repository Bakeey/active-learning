function p = plot_all...
              (reference_state_trajectory, ...
               initial_state_trajectory, initial_input_trajectory, ...
               optimal_state_trajectory, optimal_input_trajectory, fig_idx)
%PLOT_ALL plots all relevant data and does some nice post-processing
    if nargin < 6
        fig_idx = 101;
    end
    p = figure(fig_idx);
    N = size(optimal_state_trajectory,2);

    reference_plotting = zeros(4,N);
    for idx = 1:N
        reference_plotting(:,idx) = [reference_state_trajectory{idx}.x;...
                                    reference_state_trajectory{idx}.y;...
                                    reference_state_trajectory{idx}.theta;...
                                    reference_state_trajectory{idx}.time];
    end
    
    initial_plotting = zeros(6,N);
    for idx = 1:N
        initial_plotting(:,idx) = [initial_state_trajectory{idx}.x;...
                                   initial_state_trajectory{idx}.y;...
                                   initial_state_trajectory{idx}.theta;...
                                   initial_state_trajectory{idx}.time;...
                                   initial_input_trajectory(1,idx);...
                                   initial_input_trajectory(2,idx)];
    end

    optimal_plotting = zeros(6,N);
    for idx = 1:N
        optimal_plotting(:,idx) = [optimal_state_trajectory{idx}.x;...
                                   optimal_state_trajectory{idx}.y;...
                                   optimal_state_trajectory{idx}.theta;...
                                   optimal_state_trajectory{idx}.time;...
                                   optimal_input_trajectory(1,idx);...
                                   optimal_input_trajectory(2,idx)];
    end
    
    subplot(4,1,1); hold on;
    plot(reference_plotting(1,:),reference_plotting(2,:),'k--');
    plot(initial_plotting(1,:),initial_plotting(2,:),'k-.');
    plot(optimal_plotting(1,:),optimal_plotting(2,:),'k-');
    ylabel('$y$','Interpreter','latex');
    xlabel('$x$','Interpreter','latex'); hold off;

    subplot(4,1,2); hold on;
    plot(reference_plotting(4,:),reference_plotting(3,:),'k--');
    plot(initial_plotting(4,:),initial_plotting(3,:),'k:');
    plot(optimal_plotting(4,:),optimal_plotting(3,:),'k-');
    ylabel('Heading $\theta$ [Rad]','Interpreter','latex');
    xlabel('Time t[s]','Interpreter','latex'); hold off;

    subplot(4,1,3);
    hold on
    plot(optimal_plotting(4,:),optimal_plotting(1,:),'k-');
    plot(optimal_plotting(4,:),optimal_plotting(2,:),'r-');
    plot(initial_plotting(4,:),initial_plotting(1,:),'k:');
    plot(initial_plotting(4,:),initial_plotting(2,:),'r:');
    legend('x','y','Location','northwest','Interpreter','latex')
    xlabel('time [s]','Interpreter','latex')
    hold off
    
    subplot(4,1,4);
    hold on
    stairs(1:N,optimal_plotting(5,:),'k-');
    stairs(1:N,optimal_plotting(6,:),'r-');
    stairs(1:N,initial_plotting(5,:),'k:');
    stairs(1:N,initial_plotting(6,:),'r:');
    legend('$U_1$','$U_2$','Location','northwest','Interpreter','latex')
    xlabel('time [s]','Interpreter','latex')
    hold off

end

