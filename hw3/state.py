import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 6
    dt: float = 0.1

    x_0: float = 0
    y_0: float = 0
    theta_0: float = np.pi/2
    u_0 = np.array([1, -.5])

    n_particles: int = 50
    cov = 0.02


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
