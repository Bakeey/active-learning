import numpy as np
from dataclasses import dataclass

from state import Params, State
from plotting import plot_current, plot_kalman

def kinv(A: any) -> any:
    """A better inverse function"""
    return np.reciprocal(A) if isinstance(A, int) or np.size(A)==1 else np.linalg.inv(A)

def kalman(state_trajectory: np.ndarray[State],\
           measurement_trajectory: np.ndarray[float], gain: float = 1.0) -> tuple:
    # preallocation
    mean_prediction = np.zeros((np.size(state_trajectory), 2), dtype=float)
    mean_prediction[0] = measurement_trajectory[0] # MEASUREMENT OR STATE?
    mean_prediction_bar = np.zeros_like(mean_prediction)

    covariance_prediction = np.zeros((np.size(state_trajectory), 2,2), dtype=float)
    covariance_prediction_bar = np.zeros_like(covariance_prediction)

    initial_state_covariance = Params.cov  # initial covariance?
    covariance_prediction[0] = initial_state_covariance * np.eye((2),dtype=float)
    K = np.zeros_like(covariance_prediction)

    cov = Params.cov
    R = np.diag([cov, cov]) # process noise covariance
    Q = R # cov # measurement noise covariance
    A = np.eye(2, dtype=float) + Params.dt * np.array([[0, 1],[-1, 0]])
    C = np.eye(2, dtype=float) # np.array([1,0]) # sensor dynamics

    for idx in range(1,np.size(state_trajectory)):
        mean_prediction_bar[idx] = A.dot(mean_prediction[idx-1]) # State(mean_prediction[idx-1]).next()()
        covariance_prediction_bar[idx] = A.dot(covariance_prediction[idx-1]).dot(A.transpose()) + R

        # Kalman Gain
        K[idx] = gain * ( covariance_prediction_bar[idx].dot(C).dot(\
            kinv(C.dot(covariance_prediction_bar[idx]).dot(C.transpose())+Q)) )
        
        mean_prediction[idx] = mean_prediction_bar[idx] + \
            K[idx].dot(measurement_trajectory[idx] - C.dot(mean_prediction_bar[idx]))
        covariance_prediction[idx] = (np.eye(C.shape[0]) - np.multiply(C.reshape(2,2), K[idx])).dot(covariance_prediction_bar[idx])

    mean_state_prediction = np.empty_like(state_trajectory)

    for idx in range(state_trajectory.size):
        mean_state_prediction[idx] = State(mean_prediction[idx], idx*Params.dt)

    return mean_state_prediction, covariance_prediction

def main():
    T: float = Params.T
    dt: float = Params.dt
    N = int(np.ceil(T/dt))+1

    # Set up trajectory
    X_0 = State((Params.x_0, Params.y_0), t=0)
    state_trajectory: np.ndarray[State] = np.empty(N, dtype=State)
    state_trajectory[0] = X_0
    measurement_trajectory = np.zeros((state_trajectory.shape[0],2))
    measurement_covariance = np.diag([Params.cov, Params.cov])
    measurement_trajectory[0] = np.random.multivariate_normal(X_0(), measurement_covariance) # OR IS THIS WITHOUT RANDOM INIT ERROR
    
    for ii in range(N-1):
        state_trajectory[ii+1] = state_trajectory[ii].next(dt)
        measurement_trajectory[ii+1] = np.random.multivariate_normal(state_trajectory[ii+1](), measurement_covariance) # state_trajectory[ii+1]() + np.random.normal(0,np.sqrt(0.1),size=(1,2))# 

    np.random.normal()
    # plot nominal trajectory
    # plot_current(state_trajectory)

    state_prediction, covariance_prediction = kalman(state_trajectory, measurement_trajectory)
    # plot_current(state_prediction)

    plot_kalman(state_trajectory, measurement_trajectory, state_prediction)

    print(covariance_prediction)

    gains = np.linspace(0.9,1.1,40,endpoint=False) # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 1, 1.05, 1.1]
    error = np.zeros(len(gains))

    for idx, gain in enumerate(gains):
        kalman_trajectory, _ = kalman(state_trajectory, measurement_trajectory, gain)
        # plot_kalman(state_trajectory, measurement_trajectory, kalman_trajectory)
        for n in range(100):
            measurement_trajectory_random = np.empty_like(measurement_trajectory)
            for ii in range(N):
                measurement_trajectory_random[ii] = np.random.multivariate_normal(state_trajectory[ii](), measurement_covariance)
            kalman_trajectory, _ = kalman(state_trajectory, measurement_trajectory_random, gain)
            # plot_kalman(state_trajectory, measurement_trajectory_random, kalman_trajectory)
            for ii in range(N):
                error[idx] += np.sum(np.abs(kalman_trajectory[ii]() - state_trajectory[ii]())**2,axis=-1)**0.5 # np.linalg.norm(kalman_trajectory[ii]() - state_trajectory[ii]())
        error[idx] /= (100*N)

    print(gains)
    print(error)

    import matplotlib.pyplot as plt
    plt.plot(gains, error)
    plt.show()

    return

if __name__ == "__main__":
    exit(main())