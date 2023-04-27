import numpy as np
from dataclasses import dataclass

from state import Params, State
from plotting import plot_current, plot_noise

def main() -> int:
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))

    # Set up trajectory
    X_0 = State((Params.x_0, Params.y_0, Params.theta_0), t=0)
    U_0 = np.kron( np.ones((N,1)), Params.u_0 )
    input_trajectory = U_0
    state_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
    state_trajectory[0] = X_0
    for ii in range(N-1):
        state_trajectory[ii+1] = state_trajectory[ii].next(input_trajectory[ii], dt)

    # Noisy particles
    n_particles = Params.n_particles
    covariance = Params.cov
    noisy_trajectory = np.empty((N,n_particles,2),dtype=float)
    
    for idx in range(N):
        mean = (state_trajectory[idx].x, state_trajectory[idx].y)
        cov = np.array([[covariance, 0], [0, covariance]])
        noisy_trajectory[idx] = np.random.multivariate_normal(mean, cov, (n_particles,))

    plot_noise(state_trajectory, input_trajectory, noisy_trajectory)






    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)
