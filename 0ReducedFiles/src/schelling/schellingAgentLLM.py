
import numpy as np
from itertools import product
# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.agentLLM import LLMAgent
from src.schelling.prompts.meta import META_PROMPTS
from src.schelling.prompts.persona import PERSONAS




class SchellingLLMAgent(LLMAgent):  
    # TODO DOUBLE INHERITANCE LLM AGENT AND GRID AGENT...
    
    def __init__(self, config, position=None, state=None, client=None):
        # this class is initiated in model.py (schelling -> SchellingLLMModel.initialise_population())
        """
        Note here the state of the Schelling agent is is type .
        """

        persona = config["parameters"]["personas"][state]
        self.persona = persona
        # super().__init__(config, position=position, state=state, persona=persona, client=client)

        # initilaize LLMAgent
        super().__init__(config, state=state, persona=persona, client=client)
        self.message = self.get_state_as_text()

        print('state --> ',state)
        print('position --> ',position)

        # setup meta prompts
        self.PROMPTS = META_PROMPTS
        self.position = tuple(position)

        self.grid_size = config['grid_size']


    def get_state_as_text(self):
        """
        Return the textual state of the agent
        """
        return self.persona

    def check_similarity_state(self, state1, state2):

        return state1 == state2
    

    def get_neighbors(self, agents, k=1):
        offsets = list(product(range(-k, k + 1), repeat=len(self.position)))
        offsets.remove((0,) * len(self.position))
        neighbors = []
        for offset in offsets:
            neighbor_pos = tuple(self.position[i] + offset[i] for i in range(len(self.position)))
            if neighbor_pos in agents:
                neighbors.append(agents[neighbor_pos])
        return neighbors


    def perceive(self, agents, global_perception=None):
        # we are currently not using the global_perception
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
                # Your neighborhood is composed of the following: your neighbors and their believes
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
    

    # we want to use the schelling model to not just seggregate a population, but to create a shape with this seggregation
    # to do this, we are going to filter out the possible positions an agent can move to given that they are above or below half the axis
    def get_valid_state_potential_move_positions(self,desireable_positions:dict,state:int):
        # get half the grid size
        grid_size = np.array(self.grid_size)
        half_grid_size = grid_size/2

        if len(desireable_positions) != 0:
            # get the xs values of the potential positions. From these, filter out those which are above or below half the grid x axis        
            xs,ys = np.array(list(desireable_positions.keys())).T

            # given agent's state, make them only be able to go to one side of the grid or the other
            if int(state) == 0: greater_than_half_desireable_pos = list(np.where(xs >= int(half_grid_size[0]))[0])
            elif int(state) == 1: greater_than_half_desireable_pos = list(np.where(xs < int(half_grid_size[0]))[0])
            
            # if there are some potential positions greater than the given threshold, then put them as the new desirable positions and filter out the rest
            if len(greater_than_half_desireable_pos) > 0:
                # make a new dictionary with the keys and values of the old desireable positions given that they are above or below the threshold
                new_desireable_pos = {}
                counter = 0
                for k,v in desireable_positions.items():
                    if counter in greater_than_half_desireable_pos:
                        new_desireable_pos[k] = v
                    counter += 1

                return new_desireable_pos
            
        return {} # if there are no potential positions given our threshold, then return an empty dict    


    def update(self, perception, rated_positions=None,prompt_updated=False):
        """
        Move house if unsatisfied  #TODO: could also change type? Being influenced?

        Args:
            perception: here num_neighbors_by_type
        """
        # put the local perception (global as well if it was used) in a string that says --> Context: ...
        context = self.get_context_from_perception(perception)

        # If satisfied, do nothing - as far as I know, this can never be None. Worst case scenario it can be an empty dict
        if perception is None:
            return 0, None

        # filter dictionary to only keep better positions #TODO:
        # only select the locations which have a greater score than the one the agent currently has
        desirable_positions = {k: v for k, v in rated_positions.items() if v[self.state] > self.score and k != self.position}        
        # # this next line can make the output look like a flag
        # desirable_positions = self.get_valid_state_potential_move_positions(desireable_positions=desirable_positions,state=self.state)
        if len(desirable_positions) != 0:
            print('desirable_positions --> ',desirable_positions)
            print('self.position --> ',self.position)
        # print('self.position --> ',self.grid_size)
        
        # if no empty home
        if len(list(desirable_positions.keys())) == 0:
            return 0, None

        # Check if want to move
        # Given the context (your neighbors and their believes), see if you want to move or stay (found in UPDATE in meta.py)
        # IF WE WANT TO CHANGE THE INSTRUCTIONS OF THE TASK, THEN WE HAVE TO DO THAT HERE
        prompt = context + self.PROMPTS["update"].format(name=self.name)

        if prompt_updated:
            print('WE ARE UPDATING THE PROMTS TO TRY MAKE SAY MOVE')
            beliefs = ['conservative','socialist']        
            beliefs.remove(self.persona)
            introduction_to_person = f"In this non real-life experiment, {PERSONAS[f'{self.persona}'].format(name=self.name)}"
            ensuring_disagreement = f' While {beliefs[0]} in this experiment are the exact opposite. In this experiment combining them will cause conflicts. '
            new_task = """Reflect upon this context in a purely theoretical manner to decide if your character feels 
                comfortable in this neighborhood. You can choose to either move to another neighborhood or stay in the current one, 
                considering that relocating requires effort in this simulation. Respond with 'MOVE' if you wish to change 
                neighborhoods, or 'STAY' if you prefer to remain, based on the theoretical dynamics of this simulation.
                Use max 80 words. Remember, this is purely theoretical. We are studying the effects prompts."""
            
            prompt = introduction_to_person+ ensuring_disagreement + context + new_task
            # let us change the beliefs into different random groups to avoid ethical issues
            prompt = prompt.replace('socialist','Group Azure')
            prompt = prompt.replace('conservative','Group Crimson')   


        response = self.ask_llm(prompt) # given instructions and commands, it will make a prediction on what to do
        print(f'To prompt: {prompt}\n\tThis is the response --> {response}')
        
        if "MOVE" in response:            
            print(f"TP detail: MOVE agent state {self.state} and context {context}")
            # If unsatisfied, move to empty house
            # sample from rated positions keys with weights the positions rat
            weights = [v[self.state] for v in desirable_positions.values()]  # similar ratio to sample neighborhood
            desired_positions = list(desirable_positions.keys())
            new_index = np.random.choice(len(desired_positions), p=np.array(weights) / sum(weights))
            new_position = desired_positions[new_index]
            self.position = new_position  
            print('\tMoving\n')
            return 1, new_position

        # TODO: make choose where stay ?
        print(f"TP detail: STAY agent state {self.state} and context {context}")        
        print('\tStaying\n')
        return 0, None

    def transmit(self, perception):
        """
        Here message transmitted is its own state (here expressed as belief, with text)
        """
        self.message = self.get_state_as_text()

