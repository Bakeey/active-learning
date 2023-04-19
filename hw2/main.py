import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = np.pi
    dt: float = 0.001
    Q = np.diag([1,3,4])
    R = np.diag([0.01, 0.01])
    M = np.diag([1,3,4])
    alpha: float = 0.5
    beta: float = 0.5
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
            raise ValueError(f"unsupported state format: {xytheta}")
        self.t = t

    def __call__(self):
        return np.array([self.x, self.y, self.theta])

    def dynamics(self, U: np.ndarray) -> np.ndarray:
        """Dynamics of a simple two-wheeler"""
        assert U.size == 2 , f"Input U has wrong size, got: {U.size}"
        xdot: float = np.cos( self.theta ) * U[0]
        ydot: float = np.sin( self.theta ) * U[0]
        thetadot: float = U[1]
        return np.array([xdot, ydot, thetadot])
    
    def next(self, U: np.ndarray, dt: float = Params.dt):
        """Computes next state from current state with given input U"""
        assert U.size == 2 , f"Input U has wrong size, got: {U.size}"
        next_state: np.ndarray = self() + dt * self.dynamics(U)
        return State(next_state, self.t + dt)
    

class StatePertubation(State):
    def __init__(self, xytheta, t: float = 0):
        super().__init__(xytheta, t)
    
    def dynamics(self, U: np.ndarray) -> np.ndarray:
        """Linearized pertubation dynamics in D_1f/D1_f resp. D_2f/D2_f"""
        A = np.array([0, 0, -np.sin(self.theta)*U[0],\
                      0, 0,  np.cos(self.theta)*U[0],\
                      0, 0, 0]).reshape(3,3)    # D_1f
        B = np.array([np.cos(self.theta), 0, np.sin(self.theta), 0, 0, 1]).reshape(3,2) # D_2f
        x_dot = np.dot(A, self()) + np.dot(B, U)
        return x_dot


def J(state_trajectory: list[State], input_trajectory: np.ndarray) -> float:
    """Returns the cost of a given state-input trajectory"""
    assert input_trajectory.shape == (len(state_trajectory)-1,2) ,  f"State/input has wrong size"

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


def Directional_J(state_trajectory: list[State], input_trajectory: np.ndarray,\
                  state_pertubation: list[State], input_pertubation: np.ndarray ) -> float:
    """Returns the directional derivative of the cost given an state-input trajectory and a state/input pertubation"""
    assert input_trajectory.shape == (len(state_trajectory)-1,2) ,  f"State/input has wrong size"
    assert input_trajectory.shape == input_pertubation.shape,  f"Input/pertubation has wrong size"
    assert len(state_trajectory) == len(state_pertubation) ,  f"State/pertubation has wrong size"

    dt = Params.dt
    Q, R, M = Params.Q, Params.R, Params.M
    cost : float = 0

    for ii in range(len(state_trajectory)-1):
        x_curr = state_trajectory[ii]()  - np.array([2*state_trajectory[ii].t/np.pi ,0 ,np.pi/2])
        u_curr = input_trajectory[ii]
        z_curr = state_trajectory[ii]()
        v_curr = input_trajectory[ii]
        cost += dt * np.dot(z_curr,np.dot(Q, x_curr))
        cost += dt * np.dot(v_curr,np.dot(R, u_curr))
        
    x_curr = state_trajectory[-1]()
    cost += np.dot(z_curr,np.dot(M, x_curr))
    return cost


def main() -> int:
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))
    alpha = Params.alpha
    beta = Params.beta
    eps = Params.eps
    max_iterations = Params.max_iterations

    # Initial trajectory
    x_0 = 0
    y_0 = 0
    theta_0 = np.pi/2
    U_0 = np.array([1, -.5])
    initial_trajectory: list[State] = [State((x_0,y_0,theta_0),0)]
    for ii in range(N):
        initial_trajectory.append(initial_trajectory[ii].next(U_0, dt))
    # print(initial_trajectory)

    initial_cost = J(initial_trajectory, np.zeros((N,2)))
    initial_deriv = Directional_J(initial_trajectory, np.zeros((N,2)), initial_trajectory, np.zeros((N,2)))

    counter: int = 0
    while True:
        if counter > 0 and Directional_J(**args) < eps:
            break
        if counter > max_iterations:
            return 0 # no solution found
        
        




    StatePertubation((0,0,0),0).dynamics([0,0])

    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)
