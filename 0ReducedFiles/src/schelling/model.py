import itertools
import random
from src.model import GridModel
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.visualize import plot_distribution_hist
from src.utils.utils import sample_mixture_gaussian
from src.schelling.prompts.meta import META_PROMPTS
from src.schelling.schellingAgentLLM import SchellingLLMAgent



class SchellingLLMModel(GridModel):
    def __init__(self, config, id="", param_name=None):
        
        n_classes = len(config["parameters"]["personas"])
        assert n_classes == len(config["parameters"]["ratio"]), "Number of classes must be equal to the number of ratios given"

        title=f"Schelling model LLM with {n_classes} classes"
        path = "outputs/{}/".format(id)

        super().__init__(config, id=id, with_llm=True, title=title, path=path, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"
        
        n_classes = len(self.personas)

        random.seed(42) # set a seed so we ensure to get the same initial population all the time

        # create positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))        
        random.shuffle(self.positions)

        # number positions by types and some slots left empty
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        num_agents = sum(num_agents_by_type)

        states = [i for i in range(len(self.personas)) for j in range(num_agents_by_type[i])]        
        random.shuffle(states)

        print('self.positions --> ',self.positions)
        print('states --> ',states)

        #self.beliefs = self.initialise_beliefs_population(num_agents_by_type, bias, self.config["parameters_llm"]["polarization"])
        for n in range(num_agents):
            # self.client is initialised in GirdModel -> and the client is OpenAi in the case of wanting to use it
            self.agents[self.positions[n]] = SchellingLLMAgent(self.config, state=states[n], position=self.positions[n], client=self.client) 
        
        print('#########')
        print('states --> ',states)
        print('positions --> ',self.positions)
        print('agents --> ',{k:(v.state,v.position) for k,v in self.agents.items()})

    def check_similarity_state(self, state1, state2):
        return state1==state2

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        """
        # TODO: in case more than 2 types, check ratio of similar types among neighbors

        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        if len(neighbors) == 0:
            return [1 for _ in range(len(self.personas))]

        similarity_ratios = []
        for i, persona in enumerate(self.personas):
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, i)])
            similarity_ratios.append(float(count_similar) / len(neighbors))

        return similarity_ratios

    def evaluate_population(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []

        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            #should return the number of neighbors that agree with the agent's state
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
            
        return sum(similarity) / len(similarity)
    
