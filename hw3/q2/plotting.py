import matplotlib.pyplot as plt
import numpy as np

from state import State

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def plot_initial(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]
    theta = [state.theta for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]
    theta_init = [state.theta for state in initial_trajectory]

    
    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot([0,4],[0,0], 'k--', label='Reference Trajectory')
    axs[0].plot(x_init,y_init, 'k:', label='Initial Trajectory')
    axs[0].plot(x,y, 'k', label='Optimized Trajectory')
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[0].legend(loc="upper right")
    axs[1].plot([0,2*np.pi],[np.pi/2,np.pi/2], 'k--', label='Reference Trajectory')
    axs[1].plot(time, theta_init, 'k:', label='Initial Trajectory')
    axs[1].plot(time, theta, 'k', label='Optimized Trajectory')
    axs[1].set_ylabel(r'$\theta$ [rad]')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[1].set_ylim([-0.2,2.7])
    axs[1].legend(loc="upper right")
    # axs[2].plot(time, U_0[:,0], 'k:', label='Initial Trajectory')
    # axs[2].plot(time, U_0[:,0], 'k:', label='Initial Trajectory')
    # axs[2].plot([0,2*np.pi], input_trajectory[:,0], label='Optimized Trajectory')
    # axs[2].plot([0,2*np.pi], input_trajectory[:,1], label='Optimized Trajectory')
    # axs[2].set_ylabel('input magnitude')
    # axs[2].set_xlabel('time $t$ [sec]')

    plt.show()
    
    return


def plot_optimized(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]
    theta = [state.theta for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]
    theta_init = [state.theta for state in initial_trajectory]

    
    fig, axs = plt.subplots(3)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(time,x, 'k', label=r'$x$')
    axs[0].plot(time,y, 'r', label=r'$y$')
    axs[0].set_xlabel('time $t$ [sec]')
    axs[0].set_ylabel('state magnitude')
    axs[0].legend(loc="upper left")
    axs[1].plot(time, theta, 'k')
    axs[1].set_ylabel(r'$\theta$ [rad]')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[1].set_ylim([-0.2,2.7])
    axs[2].plot(time, input_trajectory[:,0], 'k', label=r'$u_0(t)$')
    axs[2].plot(time, input_trajectory[:,1], 'r', label=r'$u_1(t)$')
    axs[2].set_ylabel('input magnitude')
    axs[2].set_xlabel('time $t$ [sec]')
    axs[2].legend(loc="upper right")
    axs[2].set_ylim([-7,7])


    plt.show()
    
    return


def plot(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]
    theta = [state.theta for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]
    theta_init = [state.theta for state in initial_trajectory]

    
    fig, axs = plt.subplots(3)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x_init,y_init, label = "initial trajectory")
    axs[0].plot(x,y)
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[1].plot(time, theta_init)
    axs[1].plot(time, theta)
    axs[1].set_ylabel(r'$\theta$ [rad]')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[2].plot(time, U_0[:,0])
    axs[2].plot(time, U_0[:,0])
    axs[2].plot(time, input_trajectory[:,0])
    axs[2].plot(time, input_trajectory[:,1])
    axs[2].set_ylabel('input magnitude')
    axs[2].set_xlabel('time $t$ [sec]')

    plt.show()
    
    return

def plot_current(state_trajectory: np.ndarray[State]) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]
    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x,y)
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[1].plot(time,x, 'k', label=r'$x$')
    axs[1].plot(time,y, 'r', label=r'$y$')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[1].set_ylabel('state magnitude')
    axs[1].legend(loc="lower left")    
    plt.show()
    
    return

def plot_kalman(state_trajectory: np.ndarray[State], 
                measurements: np.ndarray, 
                kalman_trajectory: np.ndarray[State]) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]
    x_k = [state.x for state in kalman_trajectory]
    y_k = [state.y for state in kalman_trajectory]
    x_m = measurements[:,0]
    y_m = measurements[:,1]
    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x,y,'k--', label=r'actual system trajectory')
    axs[0].plot(x_k,y_k,'k', label=r'kalman filter')
    axs[0].scatter(x_m,y_m, label='measurements')
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[0].legend(loc="lower left") 
    axs[1].plot(time,x, 'k--', label=r'$x$ (actual system trajectory)')
    axs[1].plot(time,x_k, 'k', label=r'$x$ (kalman filter)')
    axs[1].plot(time,y, 'r--', label=r'$y$ (actual system trajectory)')
    axs[1].plot(time,y_k, 'r', label=r'$y$ (kalman filter)')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[1].set_ylabel('state magnitude')
    axs[1].legend(loc="lower left")    
    plt.show()
    
    return

def plot_optimality():
    K_gain = [0.8,  0.84, 0.88, 0.92, 0.96, 1.,   1.04, 1.08, 1.12, 1.16, 1.2 ]
    mean_error = [0.15626832, 0.15434268, 0.153419  , 0.1520322 , 0.15366641, 0.15154664, 0.15273871, 0.15204482 , 0.15206308, 0.1536205, 0.1588911 ]
    mean_variance = [0.0069597 , 0.00661742, 0.00679898, 0.00683991, 0.00703347, 0.0064404, 0.0072643 , 0.00697951, 0.00663712, 0.00683126, 0.00699933]
    mean_std = [np.sqrt(var) for var in mean_variance]

    fig, ax = plt.subplots()
    ax.bar(K_gain, mean_error, align='center', ecolor='black', width=0.03)
    ax.set_ylabel('mean error')
    ax.set_xticks(K_gain)
    ax.set_xticklabels(K_gain)
    ax.set_ylim(0.15,0.16)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.show()
    return 0

if __name__ == "__main__":
    exit(plot_optimality())