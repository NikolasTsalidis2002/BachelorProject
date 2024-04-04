import itertools
import random
from src.model import GridModel
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.visualize import plot_distribution_hist
from src.utils.utils import sample_mixture_gaussian
from src.models.schelling.prompts.meta import META_PROMPTS
from src.models.schelling.agentLLM import SchellingLLMAgent
from src.models.schelling.agentABM import SchellingABMAgent



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

        # create positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.seed(42)
        random.shuffle(self.positions)

        # number positions by types and some slots left empty
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        num_agents = sum(num_agents_by_type)

        states = [i for i in range(len(self.personas)) for j in range(num_agents_by_type[i])]        
        random.seed(30)
        random.shuffle(states)

        #self.beliefs = self.initialise_beliefs_population(num_agents_by_type, bias, self.config["parameters_llm"]["polarization"])
        for n in range(num_agents):
            # self.client is initialised in GirdModel -> and the client is OpenAi in the case of wanting to use it
            self.agents[self.positions[n]] = SchellingLLMAgent(self.config, state=states[n], position=self.positions[n], client=self.client) 
        

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



    """     
    def initialise_beliefs_population(self, num_agents_by_type, bias, polarization):
        
        #TODO: CASE MORE THAN 1 dimension
        #TODO Implement better initialisation; with bias shifting setup rather

        num_agents = sum(num_agents_by_type)
        std = 1.0 - polarization
        mu = 3 * polarization
        beliefs_raw = sample_mixture_gaussian(bias, means=[-mu, mu], std=[std, std], n_samples=num_agents)
        print("Generated Population:", beliefs_raw)
        beliefs = [int(round(b)) for b in beliefs_raw]

        # Clip to ensure beliefs are in the valid range [-3, 3]
        # NOTE: Currently clipped assymetrically and shifted since do not want centrist #TODO: TP
        beliefs = np.clip(beliefs, -2, 3)
        beliefs = [belief - 1 if belief <= 0 else belief for belief in beliefs]

        plot_distribution_hist(
            beliefs,
            mu=[-mu, mu],
            std=[std, std],
            scale=num_agents_by_type,
            xlabel="Political Belief",
            title="Initial Distribution of Political Beliefs",
            path=f"./outputs/{self.id}/distribution_{polarization}.png",
        )

        return beliefs 
        
        """




class SchellingABMModel(GridModel):

    def __init__(self, config, id="", param_name="similarity_threshold"):

        # ABM model
        param_value = config["parameters_abm"][param_name]
        param_value = str(int(param_value * 100)) + "%"  # TODO change if change param tested
        path = "outputs/{}/p={}_it={}".format(id, param_value, config["n_iterations"])
        title = f"Schelling ABM, w/ {param_name}={param_value}."
        super().__init__(config, id=id, with_llm=False, title=title, path=path, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"

        # create positions as positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        # number positions by types:
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]

        count = 0
        # Initialise beliefs population
        for i in range(len(self.personas)):
            for j in range(num_agents_by_type[i]):
                #state is 0,1 socialist or conservative. Position is the coordinates.
                #self.agents is a dictionary, its keys are position[count] (the coordinates)
                #the value of that would be the agent object with state and position
                self.agents[self.positions[count]] = SchellingABMAgent(self.config, state=i, position=self.positions[count])
                count += 1

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        #TODO: in case more than 2 types, check ratio of similar types among neighbors

        """
        #pos parameter is the empty positions
        #neighbors are the 8 points surrounding the empty position (assuming k=1)
        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        #? should never get to this point but idk
        if len(neighbors) == 0:
            return [1 for i in range(len(self.personas))]
        #finding how many of the neigghbors surrounding the empty position are from one persona or the other (conservative or socialist)    
        count_similar = [sum([1 for n in neighbors if n.state == i]) for i in range(len(self.personas))]
        ratios = [float(count_similar[i]) / len(neighbors) for i in range(len(self.personas))]
        return ratios  #example empty spot has 3 socialist and 5 conservative neighbors, this returns [3/8, 5/8]
       
    
    def check_similarity_state(self, state1, state2):
        return state1==state2
    
    def evaluate_population(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []
        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
        return sum(similarity) / len(similarity)


