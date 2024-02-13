
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.agentLLM import GridLLMAgent
from src.models.belief.prompts.meta import META_PROMPTS
from src.models.belief.prompts.persona import CHARACTER, STATE, MESSAGES
import yaml



class BeliefLLMAgent(GridLLMAgent):

    def __init__(self, config, position=None, state=None, extra_prompt="", persona="", message="", client=None):
        """
        Note here the state of the Schelling agent is is type .
        """

        super().__init__(config, position=position, state=state, persona=persona, extra_prompt=extra_prompt, message=message, client=client)
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
