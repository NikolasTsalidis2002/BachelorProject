import matplotlib.pyplot as plt
import itertools
import random
from src.model import GridModel
from src.agent import GridAgent
import numpy as np

# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.utils.utils import pdf, sample_mixture_gaussian
from src.agentLLM import GridLLMAgent
from src.models.belief.prompts.meta import META_PROMPTS
from src.models.belief.prompts.persona import CHARACTER, STATE, MESSAGES


class BeliefLLMModel(GridModel):
    def __init__(self, config, id="", dynamic=False):

        param_name = config["focus_parameter_llm"]
        print(config["parameters_llm"][param_name])
        param_value = "_".join([str(p) for p in config["parameters_llm"][param_name]])
        path = "outputs/{}/p={}_it={}".format(id, param_value, config["n_iterations"])
        title = f"Belief Propagation LLM, w/ {param_name}={param_value}."

        # NOTE: not dynamic, ie do not move on grid
        super().__init__(config, id=id, with_llm=True, title=title, path=path, dynamic=dynamic)

    def initialise_population(self):
        """
        A grid is initialized with a certain number of agents having different belief related to a topic.
        Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"

        # create positions as positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        # number positions by types:
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        self.n_empty = len(self.positions) - sum(num_agents_by_type)
        self.empty_positions = self.positions[: self.n_empty]
        count = self.n_empty

        for i in range(len(self.personas)):
            for j in range(num_agents_by_type[i]):
                persona = self.personas[i]
                extra_prompt = random.choice(CHARACTER[persona])
                state = random.choice(STATE[persona])
                message = random.choice(MESSAGES[persona])
                self.agents[self.positions[count]] = BeliefLLMAgent(
                    self.config, state=state, position=self.positions[count], persona=persona, extra_prompt=extra_prompt, message=message
                )
                count += 1


class BeliefLLMAgent(GridLLMAgent):

    def __init__(self, config, position=None, state=None, extra_prompt="", persona="", message=""):
        """
        Note here the state of the Schelling agent is is type .
        """

        super().__init__(config, position=position, state=state, persona=persona, extra_prompt=extra_prompt, message=message)
        self.PROMPTS = META_PROMPTS

    def get_state_numerical(self, state):
        """
        Return an int index for the state of the agent (needed only for the visualisation)
        #TODO: Should make your own !
        """
        # return index of self.persona
        return self.personas.index(self.persona)


##### ABM MODEL


# Epidemic Models:
# SI Model (Susceptible-Infected): Here, nodes (individuals) are either "susceptible" to a belief or "infected" by it. Once a node is infected, it stays infected.
# SIS Model (Susceptible-Infected-Susceptible): After a node is infected, it can become susceptible again.
# SIR Model (Susceptible-Infected-Recovered): Nodes move from being susceptible to infected and then to recovered, where they are immune to future infections.


# DeGroot Model:
# A model of consensus formation where individuals continuously update their beliefs by taking weighted averages of their neighbors' beliefs.

# Independent Cascade Model:
# Each newly activated node has a single chance to activate each of its neighbors with some probability.

# Linear Threshold Model:
# A node becomes activated if the fraction of its active neighbors exceeds a certain threshold.

# Bond Percolation:
# A model where links (or edges) in a network are activated with some probability, leading to the spread of information or behavior through the network.

# Opinion Dynamics Models:
# Voter Model: Individuals adopt the opinion of a randomly chosen neighbor.
# Axelrod's Culture Model: A model where agents with multiple cultural traits interact and influence one another.
# Bounded Confidence Models: Individuals only interact and influence each other if their opinions are sufficiently close.

# Rumor Spreading Models:
# Capture the dynamics of how rumors or information spread in a network, taking into account factors like forgetfulness and stifling of the rumor.
