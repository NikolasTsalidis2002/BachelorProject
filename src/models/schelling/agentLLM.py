
import numpy as np
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.agentLLM import GridLLMAgent
from src.models.schelling.prompts.meta import META_PROMPTS




class SchellingLLMAgent(GridLLMAgent):    
    # TODO DOUBLE INHERITANCE LLM AGENT AND GRID AGENT...
    
    def __init__(self, config, position=None, state=None, client=None):
        # this class is initiated in model.py (schelling -> SchellingLLMModel.initialise_population())
        """
        Note here the state of the Schelling agent is is type .
        """

        persona = config["parameters"]["personas"][state]
        super().__init__(config, position=position, state=state, persona=persona, client=client)
        self.message = self.get_state_as_text()

        # setup meta prompts
        self.PROMPTS = META_PROMPTS

    def get_state_as_text(self):
        """
        Return the textual state of the agent
        """
        return self.persona

    def check_similarity_state(self, state1, state2):

        return state1 == state2
    

    def perceive(self, agents, global_perception=None):
        """
        Perception by default is made by one own state, some global perception and some local perceptions (neighbor messages)
        """

        prompts = self.PROMPTS["perception"]

        perception = {}
        # list of agents that are neighbors with the agent in the update() method 
        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])

        perception["local"] = ""
        if len(neighbors) > 0:
            shared = ""
            for n in neighbors: # iterate thorugh the agent neighbors
                if n.message is not None and n.message != "":
                    # n.message = random & n.message = persona (conservative or socialist)
                    shared += prompts["local_neighbors"].format(name=n.name, message=n.message)
                    # shared is going to be a long string where it has the different names saying what their persona is (ie: james is conservative. Peter is socialist...)
            if shared != "":
                # Your neighborhood is composed of the following: shared
                perception["local"] = prompts["local"].format(local_perception=shared) + shared

        # we are not taking global perception into consideration??? -> global_perception is not given as an argument in the update method
        perception["global"] = ""
        if (global_perception is not None) and "global" in prompts.keys():
            perception["global"] = prompts["global"].format(global_perception=global_perception)

        # Update score perception
        self.score = (
            1 if len(neighbors) == 0 else sum([1 for n in neighbors if self.check_similarity_state(n.state, self.state)]) / len(neighbors)
        ) # is the ratio of neighbors that agree in persona with the agent at the time 
        return perception

    def update(self, perception, rated_positions=None):
        """
        Move house if unsatisfied  #TODO: could also change type? Being influenced?

        Args:
            perception: here num_neighbors_by_type
        """
        context = self.get_context_from_perception(perception)

        # If satisfied, do nothing
        if perception is None: # how can it be none???? how does perception represent satisfaction???
            return 0, None

        # filter dictionary to only keep better positions #TODO:
        desirable_positions = {k: v for k, v in rated_positions.items() if v[self.state] > self.score and k != self.position}

        # if no empty home
        if len(list(desirable_positions.keys())) == 0:
            return 0, None

        # Check if want to move
        # Context: perceptions[local] + update part in meta.py (conisderation part )
        prompt = context + self.PROMPTS["update"].format(name=self.name)
        response = self.ask_llm(prompt, max_tokens=5) # given instructions and commands, it will make a prediction on what to do
        if "STAY" in response:
            print(f"TP detail: STAY agent state {self.state} and context {context}")
            return 0, None

        # TODO: make choose where stay ?
        print(f"TP detail: MOVE agent state {self.state} and context {context}")

        # If unsatisfied, move to empty house
        # sample from rated positions keys with weights the positions rat
        weights = [v[self.state] for v in desirable_positions.values()]  # similar ratio to sample neighborhood
        desired_positions = list(desirable_positions.keys())
        new_index = np.random.choice(len(desired_positions), p=np.array(weights) / sum(weights))
        new_position = desired_positions[new_index]
        self.position = new_position
        
        return 1, new_position

    def transmit(self, perception):
        """
        Here message transmitted is its own state (here expressed as belief, with text)
        """
        self.message = self.get_state_as_text()

