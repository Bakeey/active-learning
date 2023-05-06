import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Essential parameters for given problem"""
    T: float = 1
    dt: float = 0.1

    x_0: float = 1.0
    y_0: float = 1.0

    n_particles: int = 100
    cov = 0.1


class State:
    def __init__(self, xy, t: float = 0):
        if isinstance(xy, np.ndarray):
            assert xy.size == 2 , f"State has wrong size, expected 2, got: {xy.size}"
            xy = xy.reshape(2)
            self.x = xy[0]
            self.y = xy[1]
        elif isinstance(xy, list) or isinstance(xy, tuple):
            self.x = xy[0]
            self.y = xy[1]
        else:
            raise ValueError(f"unsupported state format: {type(xy)}")
        self.t = t

    def __call__(self):
        return np.array([self.x, self.y])

    def dynamics(self) -> np.ndarray:
        """Dynamics of a simple two-wheeler"""
        xdot: float = self.y
        ydot: float = -self.x
        return np.array([xdot, ydot])
    
    def next(self, dt: float = Params.dt):
        """Computes next state from current state using Euler integration"""
        next_state: np.ndarray = self() + dt * self.dynamics()
        return State(next_state, self.t + dt)