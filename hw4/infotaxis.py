import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Infotaxis:
    def __init__(self, size: int = 25) -> None:
        self.size = size

        # initialize the grid (ground truth of measurement probs)
        self.grid = np.ones((size,size), dtype=float) / 100
        self.source = tuple(np.random.randint(3,22,size=(2,)))
        for idx in [-1,1]:
            self.grid[self.source] = 1
            self.grid[self.source[0] + idx*1, (self.source[1]-1):(self.source[1]+2)] = 1/2.0
            self.grid[self.source[0] + idx*2, (self.source[1]-2):(self.source[1]+3)] = 1/3.0
            self.grid[self.source[0] + idx*3, (self.source[1]-3):(self.source[1]+4)] = 1/4.0

        # plot ground truth
        fig, ax = plt.subplots(figsize=(9, 6))        
        sns.heatmap(self.grid, annot = False, cmap='Greens', linewidths=.5, linecolor = 'white')
        fig.show()

        # initialize own position
        self.initial_position = self.source
        while self.initial_position == self.source: # initialize at other position than door
            self.initial_position = tuple(np.random.randint(3,22,size=(2,)))
        self.position = self.initial_position

        # implement memory of position for plotting
        self.memory = [self.initial_position]

        # initialize useful variables
        self.prior = np.ones((size,size), dtype=float) / size**2 # prior for door l'hood
        self.likelihood = self.prior.copy()
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

    def done(self): 
        # Returns if we have found the door yet
        return self.source in self.memory[:-1] or len(self.memory) > 5000
    
    def get_position(self):
        return self.position
    
    def get_actions(self, position: tuple = None):
        """Returns available actions at a point"""
        if position is None:
            position = self.position
        _position = np.array(position)

        available_actions = []
        for idx in range(4):
            if np.isin(_position + self._action_to_direction[idx], np.arange(self.size)).all():
                available_actions.append(self._action_to_direction[idx])
        
        next_positions = [tuple(_position + action) for action in available_actions]        
        return available_actions, next_positions

    def take_measurement(self, position: tuple = None) -> bool:
        if position is None:
            position = self.position
        return np.random.random() < self.grid[position]
    
    def posterior_update(self, measured_z: bool, position: tuple = None, 
                         hypothetical: bool = False) -> None:
        if position is None:
            position = self.position

        # initialize the l'hood function
        likelihood = np.ones_like(self.grid) * (1/100 if measured_z else 99/100)
        for idx in [-1,1]:
            likelihood[position] = 1 if measured_z else 0
            try: # TODO: clipping if position is out of bounds?
                likelihood[position[0] + idx*1, (position[1]-1):(position[1]+2)] = 1/2.0
                likelihood[position[0] + idx*2, (position[1]-2):(position[1]+3)] = 1/3.0 if measured_z else 2/3.0
                likelihood[position[0] + idx*3, (position[1]-3):(position[1]+4)] = 1/4.0 if measured_z else 3/4.0
            except:
                pass
        posterior = self.prior * likelihood
        if not hypothetical:
            self.likelihood = likelihood
            self.posterior = posterior / np.sum(posterior)
            self.posterior[self.posterior < 1e-8] = 1e-8 # avoid infinite entropy
            self.entropy_posterior = -np.log(self.posterior)
            self.prior = self.posterior
        else:
            posterior = posterior / np.sum(posterior)
            posterior[posterior < 1e-8] = 1e-8 # mathematicians hate this trick
            entropy_posterior = -np.log(posterior)
            _entropy = entropy_posterior[position]
            return self.entropy_posterior[position] - _entropy

    def choose_action(self):
        """chooses the action which gives maximum entropy reduction,
        respectively minimum entropy increase"""
        available_actions, next_positions = self.get_actions(self.position)
        entropy_reduction = np.zeros(len(next_positions))

        for idx,next_position in enumerate(next_positions):
            entropy_reduction[idx] =  (1-self.likelihood[next_position]) * \
                self.posterior_update(measured_z = True, position = next_position, hypothetical=True) +\
                (self.likelihood[next_position]) * \
                self.posterior_update(measured_z = False, position = next_position, hypothetical=True
            )
        
        return available_actions[np.argmin(entropy_reduction)]
    
    def take_step(self, action):
        current_position = np.array(self.position)
        self.position = tuple( current_position + action )
        self.memory.append(self.position)

    def plot(self):
        plt.figure()
        sns.heatmap(self.grid)
        plt.show()
        plt.figure()
        sns.heatmap(self.posterior)
        memory = [(mem[1]+.5, mem[0]+.5) for mem in self.memory]
        plt.plot(*zip(*(memory[:-1])), 'b')
        plt.show()
        return

def main():
    agent = Infotaxis()
    goal = agent.source

    while not agent.done():
        position = agent.get_position()
        measurement = agent.take_measurement(position)
        agent.posterior_update(measurement, position)
        action = agent.choose_action() # TODO
        agent.take_step(action)

    agent.plot()
    return

if __name__=='__main__':
    exit(main())