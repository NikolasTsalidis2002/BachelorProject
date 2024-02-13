import matplotlib.pyplot as plt
import itertools
import random
from src.model import GridModel
from src.agentABM import GridABMAgent
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.models.schelling.prompts.meta import META_PROMPTS



class SchellingABMAgent(GridABMAgent):

    def __init__(self, config, position=None, state=None):
        """
        Note here the state of the Schelling agent is is type .
        """
        super().__init__(config, position=position, state=state)

        for key, val in config["parameters_abm"].items():
            setattr(self, key, val)

    def check_similarity_state(self, state1, state2):
        return state1 == state2

    def perceive(self, agents): 
        #if returns 0 then no neighbors
        #if returns true then unsatisfied (will move then, if satisfaction (ratio of neighbors that agree) < 0.3 which is similarity threshold)
        #if returns false then satisfied
        """
        Return the satisfaction score from its neighbors

        """
        # get the surrounding coordinates of the agent where there is an agent present 
        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])

        # if no neighbors, agent is happy and do not move ? #TODO
        if len(neighbors) == 0:
            return 0

        # Check ratio of similar types among neighbors if below threshold, unsatisfied
        # count the states of the neighbours which are the same as the agent we are currently looing at
        count_similar = sum([1 if n.state == self.state else 0 for n in neighbors]) 
        # get the ratio of neighbours which are the same as the as the agent at hand
        self.score = float(count_similar / len(neighbors))
        # return True if the percentage of different agents is less than the threshold (default 0.3 -> find in confi file)
        unsatisfied = self.score < self.similarity_threshold

        return unsatisfied

    def update(self, perception, rated_positions=None):
        """
        Move the agent to a new position if unsatisfied, based on n-dimensional space.
        """
        if perception == 0:
            return 0, None #has no neighbors, so doesnt update
        #dictionary of all empty positions that have a better persona ratio than the current satisfaction
        desirable_positions = {k: v for k, v in rated_positions.items() if v[self.state] > self.score}

        # If no desirable positions available
        if not desirable_positions:
            return 0, None

        # Choose a new position based on desirability weights
        weights = [v[self.state] for v in desirable_positions.values()]
        positions = list(desirable_positions.keys())
        #randomly chooses the next position, with higher probability of choosing the ones with better scores (aka empty positions with neighbors most similar to the current state)
        new_index = np.random.choice(len(positions), p=np.array(weights) / sum(weights))
        new_position = positions[new_index]

        # Update the agent's position to the new selected position in n-dimensional space
        self.position = new_position

        return 1, new_position
