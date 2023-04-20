import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 2*np.pi
    dt: float = 0.001

    x_0: float = 0
    y_0: float = 0
    theta_0: float = np.pi/2
    u_0 = np.array([1, -.5])

    Q = np.diag([1,3,4])
    R = np.diag([0.01, 0.01])
    M = np.diag([0,0,0]) # no terminal cost

    alpha: float = 0.5
    beta: float = 0.1
    eps: float = 0.1
    max_iterations: int = 10000


class State:
    def __init__(self, xytheta, t: float = 0):
        if isinstance(xytheta, np.ndarray):
            assert xytheta.size == 3 , f"State has wrong size, expected 3, got: {xytheta.size}"
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
        
    x_curr = state_trajectory[-1]()
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
        
    x_curr = state_trajectory[-1]()
    z_curr = state_pertubation[ii]()
    cost += np.dot(z_curr,np.dot(M, x_curr))
    return cost


def D1_l(x_curr: State) -> np.ndarray:
    x_curr = x_curr()  - np.array([2*x_curr.t/np.pi ,0 ,np.pi/2])
    Q = Params.Q
    return np.dot(Q, x_curr).reshape(3,1)


def D2_l(u_curr: np.ndarray) -> np.ndarray:
    R = Params.R
    return np.dot(R, u_curr).reshape(2,1)


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
    R_inv = np.linalg.inv(R)
    N = len(state_trajectory)
    P = np.zeros((N,3,3))
    r = np.zeros((N,3,1))
    A = np.empty((N,3,3))
    B = np.empty((N,3,2))

    # load A,B
    for ii in range(N):
        A[ii] = D1_f(state_trajectory[ii],input_trajectory[ii])
        B[ii] = D2_f(state_trajectory[ii])
    
    # iterate through P
    for ii in range(N-1,0,-1):
        x_curr = state_trajectory[ii]
        u_curr = input_trajectory[ii] # TODO make input trajectory same length as state trajectory so P(T) matches!!! [ii]
        minus_Pdot = P[ii].dot(A[ii]) + np.transpose(A[ii]).dot(P[ii]) - \
                     P[ii].dot(B[ii]).dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) + Q
        minus_rdot = np.transpose( A[ii] - B[ii].dot(R_inv).dot(np.transpose(B[ii]).dot(P[ii])) ).dot(r[ii]) +\
                     D1_l(x_curr) - P[ii].dot(B[ii]).dot(R_inv.dot(D2_l(u_curr)))
        P[ii-1] = P[ii] + dt * minus_Pdot
        r[ii-1] = r[ii] + dt * minus_rdot

    # initialize z[0] = 0 and use class StatePertubation
    state_pertubation: np.ndarray[StatePertubation] = np.empty(N, dtype=StatePertubation)
    input_pertubation: np.ndarray = np.empty_like(input_trajectory)
    state_pertubation[0] = StatePertubation((0, 0, 0), t=0)

    for ii in range(N-1):
        input_pertubation[ii] = -R_inv.dot(np.transpose(B[ii]).dot(P[ii])).dot(state_pertubation[ii]()) -\
                                 R_inv.dot(np.transpose(B[ii]).dot(r[ii])).reshape(2) -\
                                 R_inv.dot(D2_l(input_trajectory[ii])).reshape(2)
        state_pertubation[ii+1] = state_pertubation[ii].next(input_pertubation[ii], dt)

    input_pertubation[-1] = -R_inv.dot(np.transpose(B[-1]).dot(P[-1])).dot(state_pertubation[-1]()) -\
                             R_inv.dot(np.transpose(B[-1]).dot(r[-1])).reshape(2) -\
                             R_inv.dot(D2_l(input_trajectory[-1])).reshape(2)

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

    initial_cost = J(initial_trajectory, np.zeros((N,2)))
    initial_deriv = Directional_J(initial_trajectory, np.zeros((N,2)), initial_trajectory, np.zeros((N,2)))

    current_state_trajectory = initial_trajectory
    current_input_trajectory = U_0

    counter: int = 0
    while True:
        if counter > 0 and Directional_J(current_state_trajectory, current_input_trajectory, \
                                         current_state_pertubation, current_input_pertubation) <= eps:
            break
        if counter > max_iterations:
            return 0 # no solution found
        
        current_state_pertubation, current_input_pertubation = descent_direction(initial_trajectory, U_0)

        n: int = 0
        gamma: float = beta
        while True:
            if n > 0 and np.inf <= J(current_state_trajectory, current_input_trajectory) +\
                        alpha * gamma * Directional_J(current_state_trajectory, current_input_trajectory,\
                                                      current_state_pertubation, current_input_pertubation):
                break # TODO NP.INF in CONDITION
            if n > max_iterations:
                return -1 # failed to converge
            
            new_input_trajectory = current_input_trajectory + gamma * current_input_pertubation
            new_state_trajectory = X_0 + # HUH?
            n += 1
            gamma = beta**n
            # end Armijo search

        current_state_trajectory = new_state_trajectory
        current_input_trajectory = new_input_trajectory  

        counter += 1
        # end main while loop

    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)
