import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ErgodicControl:
    def __init__(self, b: float = 0.0, T: float = 100.0, dt: float = 0.1) -> None:
        self.b = b
        self.T = T
        self.dt = dt
        self.N = int(np.ceil(T/dt))

        # linear system dynamics
        self.A = np.array([[0, 1],[-1, -b]], dtype=float)
        self.X_0 = np.array([0,1], dtype=float)
        self.x = np.empty((self.N,2), dtype=float)
        self.x[0] = self.X_0

        # gaussian distribution init
        self.mu = np.array([0,0], dtype=float)
        self.Sigma = np.array([[0, 1],[-1, -b]], dtype=float)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.normalizer = ( np.linalg.det( 2*np.pi*self.Sigma ) )**-.5

        return
    
    def dynamics(self, x: np.ndarray):
        """Linear System Dynamics."""
        return self.A@x
    
    def next(self, x: np.ndarray = None):
        """Euler Integration""" # TODO: RK-4 integration if needed
        if x is None:
            x = self.X_0
        return x + self.dt * self.dynamics(x)
    
    def trajectory(self):
        for idx in range(1,self.N):
            self.x[idx] = self.next(self.x[idx-1])

        plt.plot(self.x)
        return
    
    def normal_dist(self, x: np.ndarray = None):
        if x is None:
            x = self.X_0
        assert x.shape == self.mu.shape
        prob_density: float = self.normalizer *\
            np.exp( -0.5 * np.transpose(x - self.mu)@self.Sigma_inv@(x - self.mu) )
        return prob_density



def main():
    instance = ErgodicControl()
    instance.trajectory()
    return

if __name__=='__main__':
    exit(main())