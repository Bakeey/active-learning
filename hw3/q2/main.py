import numpy as np
from dataclasses import dataclass

from state import Params, State
from plotting import plot_current, plot_kalman

def kalman(state_trajectory: np.ndarray[State],\
           measurement_trajectory: np.ndarray[float], gain: float = 1.0) -> tuple:
    # preallocation
    mean_prediction = np.zeros((np.size(state_trajectory), 2), dtype=float)
    mean_prediction[0] = state_trajectory[0]()
    mean_prediction_bar = np.zeros_like(mean_prediction)

    covariance_prediction = np.zeros((np.size(state_trajectory), 2,2), dtype=float)
    covariance_prediction_bar = np.zeros_like(covariance_prediction)

    initial_state_covariance = 0.1  # initial covariance?
    covariance_prediction[0] = initial_state_covariance * np.eye((2),dtype=float)
    K = np.zeros_like(mean_prediction)

    cov = Params.cov
    R = np.diag([cov, cov]) # process noise covariance
    Q = cov # measurement noise covariance
    A = np.eye(2, dtype=float) + Params.dt * np.array([[0, 1],[-1, 0]])
    C = np.array([1,0]) # sensor dynamics

    for idx in range(1,np.size(state_trajectory)):
        mean_prediction_bar[idx] = A.dot(mean_prediction[idx-1]) # State(mean_prediction[idx-1]).next()()
        covariance_prediction_bar[idx] = A.dot(covariance_prediction_bar[idx-1]).dot(A.transpose()) + R

        # Kalman Gain
        K[idx] = gain * ( covariance_prediction_bar[idx].dot(C).dot(\
            np.reciprocal(C.dot(covariance_prediction_bar[idx]).dot(C.transpose())+Q)) )
        
        mean_prediction[idx] = mean_prediction_bar[idx] +\
            K[idx].dot(measurement_trajectory[idx] - C.dot(mean_prediction_bar[idx]))
        covariance_prediction[idx] = (np.eye(C.shape[0]) - np.multiply(C.reshape(2,1),K[idx]).transpose()).dot(covariance_prediction_bar[idx])

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
    measurement_trajectory = np.zeros(state_trajectory.shape[0])
    measurement_trajectory[0] = Params.x_0
    measurement_covariance = Params.cov
    for ii in range(N-1):
        state_trajectory[ii+1] = state_trajectory[ii].next(dt)
        measurement_trajectory[ii+1] = np.random.normal(state_trajectory[ii+1].x, measurement_covariance)


    # plot nominal trajectory
    plot_current(state_trajectory)
    # plot_current(measurement_trajectory)

    state_prediction, covariance_prediction = kalman(state_trajectory, measurement_trajectory)
    plot_current(state_prediction)

    plot_kalman(state_trajectory, state_prediction)

    return

if __name__ == "__main__":
    exit(main())