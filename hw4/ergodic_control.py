import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import csv

from scipy.integrate import dblquad as integrate
from cachetools import cached


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class ErgodicControl:
    def __init__(self, b: float = 0.0, T: float = 1000.0, dt: float = 0.1, 
                 K: int = 10, lb: float = -10., ub: float = 10.) -> None:
        self.b = b
        self.T = T
        self.dt = dt
        self.N = int(np.ceil(T/dt))
        self.K = [(i, j) for i in range(K+1) for j in range(K+1)]

        # linear system dynamics
        self.A = np.array([[0, 1],[-1, -b]], dtype=float)
        self.X_0 = np.array([0,1], dtype=float)
        self.x = np.empty((self.N,2), dtype=float)
        self.x[0] = self.X_0

        # gaussian distribution init
        self.mu = np.array([0,0], dtype=float)
        self.Sigma = np.diag([2,2])
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.normalizer = ( np.linalg.det( 2*np.pi*self.Sigma ) )**-.5

        self.lb = lb
        self.ub = ub
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
                                    
        # plt.plot(self.x[:,0],self.x[:,-1])
        # plt.show()
        return self.x
    
    def normal_dist(self, x: np.ndarray = None):
        if x is None:
            x = self.X_0
        assert x.shape == self.mu.shape
        prob_density: float = self.normalizer *\
            np.exp( -0.5 * np.transpose(x - self.mu)@self.Sigma_inv@(x - self.mu) )
        return prob_density
        
    def normalize_basis_function(self):
        return 1 # normalization actually not that important for this case
    
    def get_basis_function(self, x: np.ndarray, k: tuple[int,int]):
        F_k = self.normalize_basis_function() # TODO
        for dim in range(x.shape[-1]):
            F_k *= np.cos( k[dim] * (x[dim]-self.ub) * np.pi / (self.lb - self.ub) ) # TODO (b-a)/(x-a)
        return F_k
        
    def get_fourier_coeffs(self):
        c_k = 1/self.T
        
        K = self.K
        coefficients = [None] * len(K)
        for idx,k in enumerate(K):
            integrator = [c_k*self.get_basis_function(_x, k) for _x in self.x]
            coefficients[idx] = np.trapz(integrator, dx=self.dt)
   
        return K, coefficients
        
    @cached(cache ={})
    def get_spatial_distro_coeffs(self):
        # TODO speed up computation significantly since this is constant for all trajectories?
        K = self.K
        l = len(K)
        coefficients = [None] * l
        for idx,k in enumerate(K):
            coefficients[idx], _ = integrate(lambda x2,x1:
                self.normal_dist(np.array([x1,x2])) * self.get_basis_function(np.array([x1,x2]),k),
                self.lb, self.ub, self.lb, self.ub)
            printProgressBar(idx + 1, l, prefix = '    Progress:', suffix = 'Complete', length = 50)
            
        return K, coefficients
    
    def get_lambda_cofficients(self):
        # TODO speed up computation significantly since this is constant for all trajectories?
        K = self.K
        l = len(K)
        coefficients = [None] * l
        s = (2+1)/2
        for idx,k in enumerate(K):
            k_squared = k[0]**2 + k[-1]**2
            coefficients[idx] = (1 + k_squared)**(-s)
            
        return K, coefficients
    

def generate_b():
    result = []
    result.extend(np.linspace(0, 1.0, num=201))  # 0.00 to 1.00, inclusive
    return result


def main():
        
    b = generate_b()
    T = [10,20,30,40,50,100,500,1000]

    ergodic_metric = [None] * len(T)
    trajectory = [None] * len(T)

    for num_T,T_i in enumerate(T):

        ergodic_metric[num_T] = [None] * len(b)
        trajectory[num_T] = [None] * len(b)

        print("Calculating Spatial Distro Coefficients")
        K, Phi_K = ErgodicControl(b=0, T=T_i).get_spatial_distro_coeffs()
        _, Lambda_K = ErgodicControl(b=0, T=T_i).get_lambda_cofficients()

        for idx, _b in enumerate(b):
            print("Calculating Ergodic Metric for b = ",_b, " and T = ", T_i)
            instance = ErgodicControl(b=_b, T=T_i)
            trajectory[num_T] = instance.trajectory()
            _, C_K = instance.get_fourier_coeffs()

            ergodic_metric[num_T][idx] = sum([ l*(c-p)**2 for l,c,p in zip(Lambda_K, C_K, Phi_K) ])
            print("    Ergodic Metric is = ",ergodic_metric[num_T][idx])

    for num_T,T_i in enumerate(T):
        label = r'$T$ = '+str(T_i)
        plt.plot(b,ergodic_metric[num_T],label=label)
    plt.xlabel(r'$b$')
    plt.ylabel(r'$\varepsilon$')
    plt.title(r'Ergodic Metric as Function of $b$ and $T$')
    plt.xlim([0,1])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Store Data
    data=[]
    for num_T,T_i in enumerate(T):
        data.append(ergodic_metric[num_T])

    # Create a CSV file
    with open("data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Value"])
        writer.writerow(["b"] + b)
        for t, values in zip(T, data):
            writer.writerow([t] + values)

    return


if __name__=='__main__':
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    exit(main())