import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from dataclasses import dataclass
from scipy.integrate import dblquad as integrate
from cachetools import cached

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 10
    dt: float = 0.01

    x_0: float = 0
    y_0: float = 1
    u_0 = np.array([1, 0])

    Q = np.diag([100,10]) # np.diag([1,3,4])
    R = np.diag([0.1,0.1]) # np.diag([0.01, 0.01])
    M = np.diag([1,0]) # no terminal cost

    alpha: float = 0.1
    beta: float = 0.5
    eps: float = 1E-4
    max_iterations: int = 10000

    K: int = 5
    lb, ub = (-5.,5.)
    mu = np.array([0,0], dtype=float)
    Sigma = np.diag([2,2])


class ErgodicControl:
    def __init__(self, T: float = Params.T, dt: float = Params.dt, 
                 K: int = Params.K, lb: float = Params.lb, ub: float = Params.ub) -> None:
        self.T = T
        self.dt = dt
        self.N = int(np.ceil(T/dt))
        self.K = [(i, j) for i in range(K+1) for j in range(K+1)]

        # linear system dynamics
        self.B = np.array([[1, 0],[0, 1]], dtype=float)
        self.X_0 = np.array([0,1], dtype=float)
        self.x = np.empty((self.N,2), dtype=float)
        self.x[0] = self.X_0

        # gaussian distribution init
        self.mu = Params.mu
        self.Sigma = Params.Sigma
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.normalizer = ( np.linalg.det( 2*np.pi*self.Sigma ) )**-.5

        self.lb = lb
        self.ub = ub
        return
    
  
#    def dynamics(self, x: np.ndarray):
#        """Linear System Dynamics."""
#        return self.B@x
#    
#    def next(self, x: np.ndarray = None):
#        """Euler Integration""" # TODO: RK-4 integration if needed
#        if x is None:
#            x = self.X_0
#        return x + self.dt * self.dynamics(x)
#    
#    def trajectory(self):
#        for idx in range(1,self.N):
#            self.x[idx] = self.next(self.x[idx-1])
#                                    
#        # plt.plot(self.x[:,0],self.x[:,-1])
#        # plt.show()
#        return self.x
    
    
    def normal_dist(self, x: np.ndarray = None):
        """Returns density of Gaussian Distribution of a given point x in R^2"""
        if x is None:
            x = self.X_0
        assert x.shape == self.mu.shape
        prob_density: float = self.normalizer *\
            np.exp( -0.5 * np.transpose(x - self.mu)@self.Sigma_inv@(x - self.mu) )
        return prob_density
        
    def normalize_basis_function(self):
        return 1 # normalization actually not that important for this case
    
    def get_basis_function(self, x: np.ndarray, k: tuple[int,int]):
        F_k = self.normalize_basis_function() # TODO
        for dim in range(x.shape[-1]):
            F_k *= np.cos( k[dim] * (x[dim]-self.ub) * np.pi / (self.lb - self.ub) )
        return F_k
        
    def get_fourier_coeffs(self):
        c_k = 1/self.T
        K = self.K
        coefficients = [None] * len(K)
        for idx,k in enumerate(K):
            integrator = [c_k*self.get_basis_function(_x, k) for _x in self.x]
            coefficients[idx] = np.trapz(integrator, dx=self.dt)
        return K, coefficients
        
    @cached(cache ={})
    def get_spatial_distro_coeffs(self):
        # TODO speed up computation significantly since this is constant for all trajectories?
        K = self.K
        l = len(K)
        coefficients = [None] * l
        for idx,k in enumerate(K):
            coefficients[idx], _ = integrate(lambda x2,x1:
                self.normal_dist(np.array([x1,x2])) * self.get_basis_function(np.array([x1,x2]),k),
                self.lb, self.ub, self.lb, self.ub)
            printProgressBar(idx + 1, l, prefix = '    Progress:', suffix = 'Complete', length = 50)
            
        return K, coefficients
    
    def get_lambda_cofficients(self):
        # TODO speed up computation significantly since this is constant for all trajectories?
        K = self.K
        l = len(K)
        coefficients = [None] * l
        s = (2+1)/2
        for idx,k in enumerate(K):
            k_squared = k[0]**2 + k[-1]**2
            coefficients[idx] = (1 + k_squared)**(-s)
            
        return K, coefficients
    


class State:
    def __init__(self, xytheta, t: float = 0):
        if isinstance(xytheta, np.ndarray):
            assert xytheta.size == 2 , f"State has wrong size, expected 2, got: {xytheta.size}"
            xytheta = xytheta.reshape(2)
            self.x = xytheta[0]
            self.y = xytheta[1]
        elif isinstance(xytheta, list) or isinstance(xytheta, tuple):
            self.x = xytheta[0]
            self.y = xytheta[1]
        else:
            raise ValueError(f"unsupported state format: {type(xytheta)}")
        self.t = t

    def __call__(self):
        return np.array([self.x, self.y])

    def dynamics(self, U: np.ndarray) -> np.ndarray:
        """Dynamics of a simple two-wheeler"""
        assert U.size == 2 , f"Input U has wrong size, expected 2, got: {U.size}"
        xdot: float = U[0]
        ydot: float = U[1]
        return np.array([xdot, ydot])
    
    def next(self, U: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U using Euler integration"""
        assert U.size == 2 , f"Input U has wrong size, expected 2, got: {U.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(U)
        return State(next_state, self.t + dt)
    

class StatePertubation(State):
    def __init__(self, xytheta, t: float = 0):
        super().__init__(xytheta, t)
    
    def dynamics(self, v: np.ndarray) -> np.ndarray:
        """Linearized pertubation dynamics in D_1f/D1_f resp. D_2f/D2_f"""
        A = np.array([0, 0,\
                      0, 0,]).reshape(2,2)    # D_1f
        B = np.array([1, 0, 0, 1]).reshape(2,2) # D_2f
        x_dot = np.dot(A, self()) + np.dot(B, v)
        return x_dot
    
    def next(self, v: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U using Euler integration"""
        assert v.size == 2 , f"Input U has wrong size, expected 2, got: {v.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(v)
        return StatePertubation(next_state, self.t + dt)


def plot_initial(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]
    
    fig, axs = plt.subplots(1)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot([0,4],[0,0], 'k--', label='Reference Trajectory')
    axs[0].plot(x_init,y_init, 'k:', label='Initial Trajectory')
    axs[0].plot(x,y, 'k', label='Optimized Trajectory')
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[0].legend(loc="upper right")
    # axs[1].plot([0,2*np.pi],[np.pi/2,np.pi/2], 'k--', label='Reference Trajectory')
    # axs[1].plot(time, theta_init, 'k:', label='Initial Trajectory')
    # axs[1].plot(time, theta, 'k', label='Optimized Trajectory')
    # axs[1].set_ylabel(r'$\theta$ [rad]')
    # axs[1].set_xlabel('time $t$ [sec]')
    # axs[1].set_ylim([-0.2,2.7])
    # axs[1].legend(loc="upper right")
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

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]

    
    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(time,x, 'k', label=r'$x$')
    axs[0].plot(time,y, 'r', label=r'$y$')
    axs[0].set_xlabel('time $t$ [sec]')
    axs[0].set_ylabel('state magnitude')
    axs[0].legend(loc="upper left")
    axs[1].plot(time, input_trajectory[:,0], 'k', label=r'$u_0(t)$')
    axs[1].plot(time, input_trajectory[:,1], 'r', label=r'$u_1(t)$')
    axs[1].set_ylabel('input magnitude')
    axs[1].set_xlabel('time $t$ [sec]')
    axs[1].legend(loc="upper right")
    axs[1].set_ylim([-7,7])


    plt.show()
    
    return


def plot(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray, initial_trajectory, U_0) -> None:
    time = [state.t for state in state_trajectory] # np.arange(0,state_trajectory.size)*Params.dt
    x = [state.x for state in state_trajectory]
    y = [state.y for state in state_trajectory]

    x_init = [state.x for state in initial_trajectory]
    y_init = [state.y for state in initial_trajectory]

    
    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    axs[0].plot(x_init,y_init, label = "initial trajectory")
    axs[0].plot(x,y)
    axs[0].set_ylabel('$y$')
    axs[0].set_xlabel('$x$')
    axs[1].plot(time, U_0[:,0])
    axs[1].plot(time, U_0[:,0])
    axs[1].plot(time, input_trajectory[:,0])
    axs[1].plot(time, input_trajectory[:,1])
    axs[1].set_ylabel('input magnitude')
    axs[1].set_xlabel('time $t$ [sec]')

    plt.show()
    
    return

def plot_P_r(P: np.ndarray, r: np.ndarray) -> None:
    P_1_1 = P[:,0,0]
    P_2_2 = P[:,1,1]
    P_1_2 = P[:,0,1]

    t = np.arange(0,P.shape[0])*Params.dt
    r_1 = r[:,0]
    r_2 = r[:,1]
    # r_3 = r[:,2] TODO is this right???

    fig, axs = plt.subplots(2)
    axs[0].plot(t,P_1_1)
    axs[0].plot(t,P_1_2)
    axs[0].plot(t,P_2_2)
    axs[1].plot(t,r_1)
    axs[1].plot(t,r_2)
    
    return


def J(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray) -> float:
    """Returns the cost of a given state-input trajectory"""
    assert input_trajectory.shape == (len(state_trajectory),2) ,  f"State/input has wrong size"

    dt = Params.dt
    Q, R, M = Params.Q, Params.R, Params.M
    cost : float = 0

    # TODO change cost function !!
    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]() # - np.array([2*state_trajectory[ii].t/np.pi ,0 ,np.pi/2])
        u_curr = input_trajectory[ii]
        cost += 0.5*dt * np.dot(x_curr,np.dot(Q, x_curr))
        cost += 0.5*dt * np.dot(u_curr,np.dot(R, u_curr))
        
    x_curr = state_trajectory[-1]() # - np.array([2*state_trajectory[-1].t/np.pi ,0 ,np.pi/2])
    cost += 0.5 * np.dot(x_curr,np.dot(M, x_curr))
    return cost


def Directional_J(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray,\
                  state_pertubation: np.ndarray[StatePertubation], input_pertubation: np.ndarray ) -> float:
    """Returns the directional derivative of the cost given an state-input trajectory and a state/input pertubation"""
    assert input_trajectory.shape == (len(state_trajectory),2) ,  f"State/input has wrong size"
    assert input_trajectory.shape == input_pertubation.shape,  f"Input/pertubation has wrong size"
    assert len(state_trajectory) == len(state_pertubation) ,  f"State/pertubation has wrong size"

    dt = Params.dt
    Q, R, M = Params.Q, Params.R, Params.M
    cost : float = 0

    # TODO chnge cost derivation!
    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]() #  - np.array([2*state_trajectory[ii].t/np.pi ,0 ,np.pi/2])
        u_curr = input_trajectory[ii]  
        z_curr = state_pertubation[ii]()
        v_curr = input_pertubation[ii]
        cost += dt * np.dot(z_curr,np.dot(Q, x_curr))
        cost += dt * np.dot(v_curr,np.dot(R, u_curr))
        
    x_curr = state_trajectory[-1]() # - np.array([2*state_trajectory[-1].t/np.pi ,0 ,np.pi/2])
    z_curr = state_pertubation[-1]()
    cost += np.dot(z_curr,np.dot(M, x_curr))
    return cost


def D1_l(x_curr: State) -> np.ndarray:
    x_curr = x_curr()  # TODO - np.array([2*x_curr.t/np.pi ,0 ,np.pi/2])
    Q = Params.Q
    return 2*np.dot(Q, x_curr).reshape(2,1)


def D2_l(u_curr: np.ndarray) -> np.ndarray:
    R = Params.R
    return 2*np.dot(R, u_curr).reshape(2,1) # TODO


def D1_f(x_curr: State, u_curr: np.ndarray) -> np.ndarray:
    matrix = np.array([0, 0,\
                      0, 0]).reshape(2,2)    # D_1f
    return matrix


def D2_f(x_curr: State) -> np.ndarray:
    matrix = np.array([1, 0, 0, 1]).reshape(2,2) # D_2f
    return matrix


def descent_direction(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray) -> tuple:
    dt = Params.dt
    Q = Params.Q
    R = Params.R
    M = Params.M
    R_inv = np.linalg.inv(R)
    N = len(state_trajectory)
    P = np.zeros((N,2,2))
    r = np.zeros((N,2,1))
    A = np.zeros((N,2,2))
    B = np.zeros((N,2,2))

    P[-1] = M

    # load A,B
    for ii in range(N):
        A[ii] = D1_f(state_trajectory[ii],input_trajectory[ii])
        B[ii] = D2_f(state_trajectory[ii])
    
    # iterate through P
    for ii in range(N-1,0,-1):
        x_curr = state_trajectory[ii]
        u_curr = input_trajectory[ii]
        minus_Pdot = P[ii].dot(A[ii]) + np.transpose(A[ii]).dot(P[ii]) - \
                     P[ii].dot(B[ii]).dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) + Q
        minus_rdot = np.transpose( A[ii] - B[ii].dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) ).dot(r[ii]) +\
                     D1_l(x_curr) - P[ii].dot(B[ii]).dot(R_inv.dot(D2_l(u_curr)))
        P[ii-1] = P[ii] + dt * minus_Pdot
        r[ii-1] = r[ii] + dt * minus_rdot

    z_0 = np.array([0,0]) # -np.linalg.inv(P[0]).dot(r[0])
    state_pertubation = np.zeros_like(state_trajectory, dtype=StatePertubation)
    input_pertubation = np.zeros_like(input_trajectory)
    state_pertubation[0] = StatePertubation(z_0, t=0)

    for jj in range(N-1):
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(P[jj]).dot(state_pertubation[jj]())).reshape(2)
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(r[jj]) + R_inv.dot(D2_l(input_trajectory[jj]))).reshape(2)
        # z_next = dt * ((A[jj].dot(state_pertubation[jj]())).reshape(3) + B[jj].dot(input_pertubation[jj]))
        z_next = state_pertubation[jj]() + dt * ((A[jj].dot(state_pertubation[jj]())).reshape(2) + B[jj].dot(input_pertubation[jj]))
        # state_pertubation[jj+1] = state_pertubation[jj].next(input_pertubation[jj])
        state_pertubation[jj+1] = StatePertubation(z_next, (jj+1)*dt)
    
    input_pertubation[-1] -= (R_inv.dot(np.transpose(B[-1])).dot(P[-1]).dot(state_pertubation[-1]())).reshape(2)
    input_pertubation[-1] -= (R_inv.dot(np.transpose(B[-1])).dot(r[-1]) + R_inv.dot(D2_l(input_trajectory[-1]))).reshape(2)

    # initialize z[0] = 0 and use class StatePertubation
    # state_pertubation: np.ndarray[StatePertubation] = np.empty(N, dtype=StatePertubation)
    # input_pertubation: np.ndarray = np.zeros_like(input_trajectory)
    # z_0 = - np.linalg.inv(P[0]).dot(r[0]).reshape(3)
    # state_pertubation[0] = StatePertubation(z_0, t=0)
# 
    # for ii in range(N-1):
    #     input_pertubation[ii] = -R_inv.dot(np.transpose(B[ii]).dot(P[ii])).dot(state_pertubation[ii]()) -\
    #                              R_inv.dot(np.transpose(B[ii]).dot(r[ii])).reshape(2) -\
    #                              R_inv.dot(D2_l(input_trajectory[ii])).reshape(2)
    #     state_pertubation[ii+1] = state_pertubation[ii].next(input_pertubation[ii], dt)
# 
    # input_pertubation[-1] = -R_inv.dot(np.transpose(B[-1]).dot(P[-1])).dot(state_pertubation[-1]()) -\
    #                          R_inv.dot(np.transpose(B[-1]).dot(r[-1])).reshape(2) -\
    #                          R_inv.dot(D2_l(input_trajectory[-1])).reshape(2)

    return (state_pertubation, input_pertubation)


def main() -> int:
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))
    alpha = Params.alpha
    beta = Params.beta
    eps = Params.eps
    max_iterations = Params.max_iterations

    # Initial trajectory
    # TODO Make some fancy Trajectory class

    X_0 = State((Params.x_0, Params.y_0), t=0)
    U_0 = np.kron( np.ones((N,1)), Params.u_0 )
    initial_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
    initial_trajectory[0] = X_0
    for ii in range(N-1):
        initial_trajectory[ii+1] = initial_trajectory[ii].next(U_0[ii], dt)

    initial_cost = J(initial_trajectory, U_0)
    initial_deriv = Directional_J(initial_trajectory, U_0, initial_trajectory, U_0)
    cost = []
    dcost = []

    current_state_trajectory = initial_trajectory
    current_input_trajectory = U_0

    z_0 = np.array([0,0])
    current_state_pertubation = np.zeros_like(current_state_trajectory, dtype=StatePertubation)
    current_input_pertubation = np.zeros_like(current_input_trajectory)
    current_state_pertubation[:] = StatePertubation(z_0, t=0)
    counter: int = 0
    while True:
        current_cost= J(current_state_trajectory, current_input_trajectory)
        current_dcost = Directional_J(current_state_trajectory, current_input_trajectory, current_state_pertubation, current_input_pertubation)
        cost.append(current_cost)
        dcost.append(current_dcost)
        print("    J = ", current_cost)
        print("    dJ = ",current_dcost)
        if counter > 0 and abs(Directional_J(current_state_trajectory, current_input_trajectory, \
                                         current_state_pertubation, current_input_pertubation)) <= eps:
            break
        if counter > max_iterations:
            return 0 # no solution found
        
        current_state_pertubation, current_input_pertubation = descent_direction(current_state_trajectory, current_input_trajectory)

        n: int = 0
        gamma: float = 1
        while True:
            new_input_trajectory = current_input_trajectory + gamma * current_input_pertubation
            new_state_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
            new_state_trajectory[0] = X_0
            for ii in range(N-1):
                new_state_trajectory[ii+1] = new_state_trajectory[ii].next(new_input_trajectory[ii], dt)

            if n > 0 and J(new_state_trajectory, new_input_trajectory) <\
                         J(current_state_trajectory, current_input_trajectory) +\
                          alpha * gamma * Directional_J(current_state_trajectory, current_input_trajectory,\
                                                      current_state_pertubation, current_input_pertubation):
                break
            if n > max_iterations:
                return -1 # failed to converge
            
            
            # new_state_trajectory = X_0 + # HUH?
            n += 1
            gamma = beta**n
            # print("    Current n:     ", n)
            # print("    Current gamma: ", gamma)
            # end Armijo search

        current_input_trajectory = new_input_trajectory
        current_state_trajectory = new_state_trajectory

        counter += 1
        print("Current counter: ", counter)
        # end main while loop
    
    fig, axs = plt.subplots(2)
    axs[0].plot(np.arange(len(cost)), cost, 'k', label=r'J$(\xi_i)$')
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel('iterations')
    # axs[0].set_ylabel(r'J$(\xi_i)$')

    axs[1].plot(np.arange(len(dcost)-1), [abs(dcost_) for dcost_ in dcost[1:]], 'k',label=r'DJ$(\xi_i)\cdot\zeta$')
    axs[1].plot([0,(len(dcost)-1)], [1E-4,1E-4], 'k--',label=r'Baseline $10^{-4}$')
    axs[1].set_yscale('log')
    axs[1].legend(loc="upper right")
    # axs[1].set_ylabel(r'DJ$(\xi_i)\cdot\zeta$')
    axs[1].set_xlabel('iterations')

    plt.show()

    plot_optimized(current_state_trajectory, current_input_trajectory, initial_trajectory, U_0)

    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)
