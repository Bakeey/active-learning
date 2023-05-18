import numpy as np

class Infotaxis:
    def __init__(self, size: int = 25) -> None:
        self.size = size

        # initialize the grid (ground truth of measurement probs)
        self.grid = np.ones((size,size), dtype=float) / size**2
        self.source = tuple(np.random.randint(3,22,size=(2,)))
        for idx in [-1,1]:
            self.grid[self.source] = 1
            self.grid[self.source[0] + idx*1, (self.source[1]-1):(self.source[1]+2)] = 1/2.0
            self.grid[self.source[0] + idx*2, (self.source[1]-2):(self.source[1]+3)] = 1/3.0
            self.grid[self.source[0] + idx*3, (self.source[1]-3):(self.source[1]+4)] = 1/4.0

        # initialize own position
        self.initial_position = self.source
        while self.initial_position == self.source: # initialize at other position than door
            self.initial_position = tuple(np.random.randint(0,25,size=(2,)))
        self.position = self.initial_position

        # implement memory of position for plotting
        self.memory = [self.initial_position]

        # initialize useful variables
        self.prior = np.ones((size,size), dtype=float) / size**2 # prior for door l'hood
        self.p_r0 = self.prior # door l'hood posterior
        self.entropy = np.ones_like(self.prior) * np.log(size**2) # prior for entropy

        """
        The following dictionary maps abstract actions to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),    # right
            1: np.array([0, 1]),    # up
            2: np.array([-1, 0]),   # left
            3: np.array([0, -1]),   # down
        }

        # posterior



        # various likelihoods

    def take_measurement(self) -> bool:
        return np.random.random() < self.grid[self.position]
    
    def posterior_update(self) -> None:
        if self.take_measurement(): # z=1
            self.likelihood_1 = np.array([1/100.0, 1/2.0, 1/100.0, 1/2.0])
            self.likelihood_0 = np.ones_like(self.likelihood_1) - self.likelihood_1
        else:
            self.likelihood_0 = np.array([99/100.0, 1/2.0, 99/100.0, 1/2.0])
            self.likelihood_1 = np.ones_like(self.likelihood_1) - self.likelihood_1
        
            






        

def main():
    test = Infotaxis(25)
    return

if __name__=='__main__':
    exit(main())