import numpy as np
from dataclasses import dataclass

from state import Params, State
from plotting import plot_current, plot_noise

def main() -> int:
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))+1

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
    cov = np.array([[covariance, 0, 0], [0, covariance, 0], [0, 0, covariance]])
    noisy_trajectory = np.empty((N,n_particles,3),dtype=float)
    noisy_mean = np.empty((N,3),dtype=float)
    noisy_mean[0] = X_0()
        
    for idx in range(N):
        mean = (state_trajectory[idx].x, state_trajectory[idx].y, state_trajectory[idx].theta)
        noisy_trajectory[idx] = np.random.multivariate_normal(mean, cov, (n_particles,))
    
    predicted_next_state = np.empty_like(noisy_trajectory)
    weight_sample = np.empty_like(noisy_trajectory)[:,:,0]

    for idx in range(1,N):
        for sample in range(n_particles):
            # sample from p(x_t|x_t-1)
            predicted_next_state[idx,sample] = State(noisy_trajectory[idx-1,sample],dt*(idx-1)).next(U_0[idx-1])()
            
        # importance sampling
        mean_prediction = np.mean(predicted_next_state[idx], axis=0)
        for sample in range(n_particles):
            sample_error = noisy_trajectory[idx,sample]-mean_prediction
            weight_sample[idx,sample] = np.exp(-sample_error.dot(cov).dot(sample_error))
        weight_sample[idx] = weight_sample[idx] / np.sum(weight_sample[idx])
        noisy_mean[idx] = weight_sample[idx].dot(noisy_trajectory[idx])


            



    plot_noise(state_trajectory, input_trajectory, noisy_trajectory, noisy_mean)
    return 1

if __name__ == "__main__":
    success = main()
    print("terminated with exit code: ", success)