import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 2*np.pi
    dt: float = 2*np.pi/1000

    x_0: float = 0
    y_0: float = 0
    theta_0: float = np.pi/2
    u_0 = np.array([1, -.5])

    Q = np.diag([100,10,10]) # np.diag([1,3,4])
    R = np.diag([0.01,0.01]) # np.diag([0.01, 0.01])
    M = np.diag([0,0,0]) # no terminal cost

    alpha: float = 0.1
    beta: float = 0.5
    eps: float = 1E-4
    max_iterations: int = 10000


class State:
    def __init__(self, xytheta, t: float = 0):
        if isinstance(xytheta, np.ndarray):
            assert xytheta.size == 3 , f"State has wrong size, expected 3, got: {xytheta.size}"
            xytheta = xytheta.reshape(3)
            self.x = xytheta[0]
            self.y = xytheta[1]
            self.theta = xytheta[2]
        elif isinstance(xytheta, list) or isinstance(xytheta, tuple):
            self.x = xytheta[0]
            self.y = xytheta[1]
            self.theta = xytheta[2]
        else:
            raise ValueError(f"unsupported state format: {type(xytheta)}")
        self.t = t

    def __call__(self):
        return np.array([self.x, self.y, self.theta])

    def dynamics(self, U: np.ndarray) -> np.ndarray:
        """Dynamics of a simple two-wheeler"""
        assert U.size == 2 , f"Input U has wrong size, expected 2, got: {U.size}"
        xdot: float = np.cos( self.theta ) * U[0]
        ydot: float = np.sin( self.theta ) * U[0]
        thetadot: float = U[1]
        return np.array([xdot, ydot, thetadot])
    
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
        A = np.array([0, 0, -np.sin(self.theta)*v[0],\
                      0, 0,  np.cos(self.theta)*v[0],\
                      0, 0, 0]).reshape(3,3)    # D_1f
        B = np.array([np.cos(self.theta), 0, np.sin(self.theta), 0, 0, 1]).reshape(3,2) # D_2f
        x_dot = np.dot(A, self()) + np.dot(B, v)
        return x_dot
    
    def next(self, v: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U using Euler integration"""
        assert v.size == 2 , f"Input U has wrong size, expected 2, got: {v.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(v)
        return StatePertubation(next_state, self.t + dt)
    

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
    axs[1].plot(time, theta_init)
    axs[1].plot(time, theta)
    axs[2].plot(time, U_0[:,0])
    axs[2].plot(time, U_0[:,0])
    axs[2].plot(time, input_trajectory[:,0])
    axs[2].plot(time, input_trajectory[:,1])
    plt.show()
    
    return

def plot_P_r(P: np.ndarray, r: np.ndarray) -> None:
    P_1_1 = P[:,0,0]
    P_2_2 = P[:,1,1]
    P_3_3 = P[:,2,2]
    P_1_2 = P[:,0,1]
    P_1_3 = P[:,0,2]
    P_2_3 = P[:,1,2]
    t = np.arange(0,P.shape[0])*Params.dt
    r_1 = r[:,0]
    r_2 = r[:,1]
    r_3 = r[:,2]

    fig, axs = plt.subplots(2)
    axs[0].plot(t,P_1_1)
    axs[0].plot(t,P_1_2)
    axs[0].plot(t,P_1_3)
    axs[0].plot(t,P_2_2)
    axs[0].plot(t,P_2_3)
    axs[0].plot(t,P_3_3)
    axs[1].plot(t,r_1)
    axs[1].plot(t,r_2)
    axs[1].plot(t,r_3)
    
    return


def J(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray) -> float:
    """Returns the cost of a given state-input trajectory"""
    assert input_trajectory.shape == (len(state_trajectory),2) ,  f"State/input has wrong size"

    dt = Params.dt
    Q, R, M = Params.Q, Params.R, Params.M
    cost : float = 0

    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]() - np.array([2*state_trajectory[ii].t/np.pi ,0 ,np.pi/2])
        u_curr = input_trajectory[ii]
        cost += 0.5*dt * np.dot(x_curr,np.dot(Q, x_curr))
        cost += 0.5*dt * np.dot(u_curr,np.dot(R, u_curr))
        
    x_curr = state_trajectory[-1]() - np.array([2*state_trajectory[-1].t/np.pi ,0 ,np.pi/2])
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

    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]()  - np.array([2*state_trajectory[ii].t/np.pi ,0 ,np.pi/2])
        u_curr = input_trajectory[ii]  
        z_curr = state_pertubation[ii]()
        v_curr = input_pertubation[ii]
        cost += dt * np.dot(z_curr,np.dot(Q, x_curr))
        cost += dt * np.dot(v_curr,np.dot(R, u_curr))
        
    x_curr = state_trajectory[-1]() - np.array([2*state_trajectory[-1].t/np.pi ,0 ,np.pi/2])
    z_curr = state_pertubation[-1]()
    cost += np.dot(z_curr,np.dot(M, x_curr))
    return cost


def D1_l(x_curr: State) -> np.ndarray:
    x_curr = x_curr()  - np.array([2*x_curr.t/np.pi ,0 ,np.pi/2])
    Q = Params.Q
    return 2*np.dot(Q, x_curr).reshape(3,1)


def D2_l(u_curr: np.ndarray) -> np.ndarray:
    R = Params.R
    return 2*np.dot(R, u_curr).reshape(2,1)


def D1_f(x_curr: State, u_curr: np.ndarray) -> np.ndarray:
    matrix = np.array([0, 0, -np.sin(x_curr.theta)*u_curr[0],\
                      0, 0,  np.cos(x_curr.theta)*u_curr[0],\
                      0, 0, 0]).reshape(3,3)    # D_1f
    return matrix


def D2_f(x_curr: State) -> np.ndarray:
    matrix = np.array([np.cos(x_curr.theta), 0, np.sin(x_curr.theta), 0, 0, 1]).reshape(3,2) # D_2f
    return matrix


def descent_direction(state_trajectory: np.ndarray[State], input_trajectory: np.ndarray) -> tuple:
    dt = Params.dt
    Q = Params.Q
    R = Params.R
    M = Params.M
    R_inv = np.linalg.inv(R)
    N = len(state_trajectory)
    P = np.zeros((N,3,3))
    r = np.zeros((N,3,1))
    A = np.zeros((N,3,3))
    B = np.zeros((N,3,2))

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

    z_0 = np.array([0,0,0]) # -np.linalg.inv(P[0]).dot(r[0])
    state_pertubation = np.zeros_like(state_trajectory, dtype=StatePertubation)
    input_pertubation = np.zeros_like(input_trajectory)
    state_pertubation[0] = StatePertubation(z_0, t=0)

    for jj in range(N-1):
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(P[jj]).dot(state_pertubation[jj]())).reshape(2)
        input_pertubation[jj] -= (R_inv.dot(np.transpose(B[jj])).dot(r[jj]) + R_inv.dot(D2_l(input_trajectory[jj]))).reshape(2)
        # z_next = dt * ((A[jj].dot(state_pertubation[jj]())).reshape(3) + B[jj].dot(input_pertubation[jj]))
        z_next = state_pertubation[jj]() + dt * ((A[jj].dot(state_pertubation[jj]())).reshape(3) + B[jj].dot(input_pertubation[jj]))
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

    X_0 = State((Params.x_0, Params.y_0, Params.theta_0), t=0)
    U_0 = np.kron( np.ones((N,1)), Params.u_0 )
    initial_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
    initial_trajectory[0] = X_0
    for ii in range(N-1):
        initial_trajectory[ii+1] = initial_trajectory[ii].next(U_0[ii], dt)

    initial_cost = J(initial_trajectory, U_0)
    initial_deriv = Directional_J(initial_trajectory, U_0, initial_trajectory, U_0)

    current_state_trajectory = initial_trajectory
    current_input_trajectory = U_0

    z_0 = np.array([0,0,0])
    current_state_pertubation = np.zeros_like(current_state_trajectory, dtype=StatePertubation)
    current_input_pertubation = np.zeros_like(current_input_trajectory)
    current_state_pertubation[:] = StatePertubation(z_0, t=0)
    counter: int = 0
    while True:
        print(Directional_J(current_state_trajectory, current_input_trajectory, current_state_pertubation, current_input_pertubation))
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
    
    plot(current_state_trajectory, current_input_trajectory, initial_trajectory, U_0)

    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)
