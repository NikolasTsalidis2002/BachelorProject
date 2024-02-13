
import random
import copy
import json
import os
from src.visualize import generate_gif_from_data_grid, plot_grid, plot_base
from src.utils.utils import json_serial
import time

### Some Models Class


class GridModel():
    
    def __init__(self, config, id="", with_llm=False, title="", path="", dynamic=True):
        
        self.id = id
        self.config=config
        self.with_llm=with_llm

        self.title=title
        self.path=path

        # Grid size
        self.dimensions = config["grid"]["dimensions"]  # Now expecting dimensions as a list for n-dimensions

        # Parameters model from config file
        for key, val in config["parameters"].items():
            setattr(self, key, val)

        if self.with_llm:
            for key, val in config["parameters_llm"].items():
                setattr(self, key, val)
        else:
            for key, val in config["parameters_abm"].items():
                setattr(self, key, val)    


        #initialise population
        self.agents = {}
        self.empty_positions = []
        self.rated_positions = {}
        self.initialise_population()

        #NOTE: dynamic model, ie agents can move on grid
        self.dynamic=dynamic
        
        self.num_agents = len(self.agents)
 
    def initialise_population(self):
        pass

    
    def evaluate_position(self, pos, k=1):
        pass


    def evaluate_empty_positions(self):
        """
        Evaluate the spot, if empty, return None
        """
        #reset
        self.rated_positions={}
        for position in self.empty_positions:
            self.rated_positions[position]=self.evaluate_position(position, k=self.perception_radius)


    def get_state_numerical(self, state):
        """
        Return an int index for the state of the agent (needed only for the visualisation)
        #TODO: Should make your own !
        """

        if type(state) == int:
            return state
        
        if type(state) == float:
            return state
        
        if type(state) == str: #NOTE: THAT IS not desirable, code your own
            return 0

    
    def update(self, **kwargs):
        """
        Update the state of all agents in the grid with a certain likelihood.
        """
        count=0
        #TODO 
        self.old_agents = copy.deepcopy(self.agents)

        #0-- rate all empty positions if dynamic
        if self.dynamic:
            self.evaluate_empty_positions()

        for agent in self.old_agents.values():
            r=random.random()
            old_position = agent.position

            #TODO: If do not update each step keep memory ?
            if r <= self.update_likelihood:
                # 1 --- perception of surrounding from previous time step
                perception=agent.perceive(self.agents)
                
                # 2--- action agent
                if_action, new_position=agent.update(perception, rated_positions=self.rated_positions, **kwargs)
                count+=if_action
                if self.dynamic and (new_position is not None):
                    self.move_agent_on_grid(agent, old_position, new_position)
                time.sleep(1)
        return count
    
    def move_agent_on_grid(self, agent,old_position, new_position):
        """
        Move agent to new position in grid if dynamic model
        """
        # Move agent to new position in grid
        self.agents[new_position] = agent
        print("Moving agent from {} to {}".format(old_position, new_position))
        del self.agents[old_position]
        self.empty_positions.remove(new_position)
        self.empty_positions.append(old_position)

        del self.rated_positions[new_position]

    def save_historics(self, output_file):
        """
        Save the historics of the agents (e.g., prompt, messages received, etc.).
        """
        historics = {}
        for pos, agent in self.agents.items():
            pos_str = "_".join(map(str, pos))  # Adjust for n-dimensional positions
            historics[pos_str] = agent.historics

        with open(output_file, "w") as f:
            json.dump(historics, f, default=json_serial)


    def evaluate_population(self):
        """
        evaluation method of the whole population according a metric tbd
        """
        return -1
    

    def run(self, n_iterations=None):
        """"
        Run the simulation for n_iterations
        Save data every X steps and do some visualisation of the results.

        """

        # For storing agent states at each iteration
        if n_iterations is None:
            n_iterations = self.config["n_iterations"]

        data = {}  
        num_actions=[]
        score_population=[]

        # 1-- Run the simulation for n_iterations
        for i in range(n_iterations):
            count=self.update()
            num_actions.append(count)
            score_population.append(self.evaluate_population())
            print("Step " + str(i) + " : " + str(count) + " updates")
    
            if count == 0 and self.early_stopping: #then stop
                print("Converged, early stopping at {} iterations".format(i))
                break

            # Save data every X steps
            if i % self.config["save_every"] == 0:
                print("TP Saving iteration " + str(i))
                data[i] = {str(key): val.state for key, val in self.agents.items()}
            

        # 2--- Plot the final state
        if not self.config["dev"]:

            #create folder if not exist
            if not os.path.exists("outputs/"+self.id):
                os.makedirs("outputs/"+self.id)

            final_score=round(self.evaluate_population(),3)

            #NOTE: if data string, convert it to float or int for visualisation
            last_data={str(key): self.get_state_numerical(val.state) for key, val in self.agents.items()}
            
            plot_grid(self.config, last_data, title=self.title+f"Final Score {final_score}", output_file=self.path+"_grid.png", with_llm=self.with_llm)
            
            #num actions plot to see evolutions
            plot_base(num_actions, y_label="Num action",x_label="iterations", output_file=self.path+"_num_actions.png")

            #score plot to see evolutions
            plot_base(score_population, y_label="Score",x_label="iterations", output_file=self.path+"_score_pop.png")

            if len(list(data.keys())) > 0: 
                with open(self.path+".json", "w") as f:
                    json.dump(data, f, default=json_serial)
                generate_gif_from_data_grid(self.config, data_file=self.path+".json",output_file=self.path+"_grid", title=self.title, with_llm=self.with_llm)
            
        return final_score
       
