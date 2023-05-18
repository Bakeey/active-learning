import numpy as np
from dataclasses import dataclass

from state import Params, State
from plotting import plot_current, plot_kalman

def kinv(A: any) -> any:
    """A better inverse function"""
    return np.reciprocal(A) if isinstance(A, int) or np.size(A)==1 else np.linalg.inv(A)


def kalman(_,measurements, gain = 1.0):
    state_estimate_bar = np.empty_like(measurements, dtype=float)
    covariance_estimate_bar = np.empty((measurements.shape[0],2,2), dtype=float)

    state_estimate = np.empty_like(state_estimate_bar)
    covariance_estimate = np.empty_like(covariance_estimate_bar)

    state_estimate[0] = measurements[0]
    state_estimate_bar[0] = measurements[0]
    
    covariance_matrix = np.diag([Params.cov, Params.cov])
    covariance_estimate[0] = covariance_matrix
    covariance_estimate_bar[0] = covariance_matrix
    Q = covariance_matrix
    R = covariance_matrix
    F = np.eye(2) + Params.dt*np.array([[0, 1],[-1, 0]])
    H = np.eye(2)

    for idx in range(1, measurements.shape[0]):
        state_estimate_bar[idx] = F.dot(state_estimate_bar[idx-1])
        covariance_estimate_bar[idx] = np.dot(np.dot(F, covariance_estimate_bar[idx-1]),np.transpose(F)) + Q

        S = covariance_estimate_bar[idx] + R
        Kalman_gain = gain * np.dot(np.dot(covariance_estimate_bar[idx],H),np.linalg.inv(S))

        state_estimate[idx] = state_estimate_bar[idx] + Kalman_gain.dot(measurements[idx] - H.dot(state_estimate_bar[idx]))
        covariance_estimate[idx] = np.dot(np.eye(2) - Kalman_gain, covariance_estimate_bar[idx])

    state_estimate_s = np.empty(measurements.shape[0], dtype=State)
    for idx in range(measurements.shape[0]):
        state_estimate_s[idx] = State(state_estimate[idx],idx*Params.dt)

    return state_estimate_s, covariance_estimate


"""
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
        mean_prediction_bar[idx] = A.dot(mean_prediction_bar[idx-1]) # State(mean_prediction[idx-1]).next()()
        covariance_prediction_bar[idx] = A.dot(covariance_prediction_bar[idx-1]).dot(A.transpose()) + R

        # Kalman Gain
        K[idx] = gain * ( covariance_prediction_bar[idx].dot(C).dot(\
            np.linalg.inv(C.dot(covariance_prediction_bar[idx]).dot(C.transpose())+Q)) )
        
        mean_prediction[idx] = mean_prediction_bar[idx] + \
            K[idx].dot(measurement_trajectory[idx] - C.dot(mean_prediction_bar[idx]))
        covariance_prediction[idx] = (np.eye(2) - K[idx] ).dot(covariance_prediction_bar[idx])

    mean_state_prediction = np.empty_like(state_trajectory)

    for idx in range(state_trajectory.size):
        mean_state_prediction[idx] = State(mean_prediction[idx], idx*Params.dt)

    return mean_state_prediction, covariance_prediction
"""
    
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
    measurement_trajectory[0] = X_0() + np.random.normal(0, Params.cov**0.5, size=(1,2))# np.random.multivariate_normal(X_0(), measurement_covariance) # OR IS THIS WITHOUT RANDOM INIT ERROR
    
    for ii in range(N-1):
        state_trajectory[ii+1] = state_trajectory[ii].next(dt)
        # measurement_trajectory[ii+1] = np.random.multivariate_normal(state_trajectory[ii+1](), measurement_covariance) # state_trajectory[ii+1]() + np.random.normal(0,np.sqrt(0.1),size=(1,2))# 
        measurement_trajectory[ii+1] = state_trajectory[ii+1]() + np.random.normal(0, Params.cov**0.5, size=(1,2))

    # plot nominal trajectory
    # plot_current(state_trajectory)

    state_prediction, covariance_prediction = kalman(state_trajectory, measurement_trajectory)
    # plot_current(state_prediction)

    plot_kalman(state_trajectory, measurement_trajectory, state_prediction)

    print(covariance_prediction)

    gains = np.linspace(0.8,1.2, 11, endpoint=True)
    error = np.zeros(len(gains))
    variance = np.zeros(len(gains))

    for idx, gain in enumerate(gains):
        state_errors = np.zeros((100,len(state_prediction)), dtype=float)
        for jdx in range(100):
            measurement_trajectory = np.zeros((len(state_trajectory),2), dtype=float)
            for t in range(np.size(state_trajectory)):
                measurement_trajectory[t] = np.random.multivariate_normal(state_trajectory[t](), measurement_covariance)
            state_prediction, _ = kalman(state_trajectory, measurement_trajectory, gain=gain)
            

            for t in range(np.size(state_trajectory)):
                state_errors[jdx,t] = np.linalg.norm( state_prediction[t]() - state_trajectory[t]() )
        error[idx] = np.mean(state_errors)
        variance[idx] = np.var(state_errors)

    print(gains)
    print(error)
    print(variance)

    import matplotlib.pyplot as plt
    plt.plot(gains, error)
    plt.show()

    """
    [0.8  0.84 0.88 0.92 0.96 1.   1.04 1.08 1.12 1.16 1.2 ]
[0.15261089 0.15817531 0.15256947 0.15505242 0.15169262 0.15230729
 0.15738326 0.15926118 0.15498004 0.15786136 0.15564271]
[0.00644538 0.00721572 0.00674815 0.00730859 0.00716775 0.00696049
 0.00682786 0.00747304 0.00650631 0.007071   0.00708925]

 [0.8  0.84 0.88 0.92 0.96 1.   1.04 1.08 1.12 1.16 1.2 ]
[0.15626832 0.153419   0.1520322  0.15366641 0.15434268 0.15154664
 0.1588911  0.15273871 0.15204482 0.1536205  0.15206308]
[0.0069597  0.00661742 0.00679898 0.00683991 0.00703347 0.0064404
 0.0072643  0.00697951 0.00663712 0.00683126 0.00699933]
    
    """

    return

if __name__ == "__main__":
    exit(main())