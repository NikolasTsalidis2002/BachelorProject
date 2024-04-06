import random
import copy
import json
import os
from src.visualize import generate_gif_from_data_grid, plot_grid, plot_base
from src.utils.utils import json_serial
import time
import yaml
import ollama
from src.schelling.prompts.persona import PERSONAS
from src.schelling.prompts.meta import META_PROMPTS



class GridModel():
    
    def __init__(self, config, id="", with_llm=True, title="", path="", dynamic=True):
        
        self.id = id
        self.config=config
        self.with_llm=with_llm

        self.title=title
        self.path=path

        # Grid size
        self.dimensions = config["grid_size"]  # Now expecting dimensions as a list for n-dimensions

        # Parameters model from config file
        for key, val in config["parameters"].items():
            setattr(self, key, val)

        if self.with_llm:
            for key, val in config["parameters_llm"].items():
                # to the class GridModel, add attributes given in config["parameters_llm"] (among them llm_name)
                setattr(self, key, val)

        self.client = None
        if with_llm and "gpt" in self.config["parameters_llm"]["llm_name"]:
            self.init_openai_client()


        #initialise population
        self.agents = {}
        self.initialise_population()

        #NOTE: dynamic model, ie agents can move on grid
        self.dynamic=dynamic
        
        self.num_agents = len(self.agents)
 

    # NOT USING THIS
    def init_openai_client(self):
        """
        Init the openai client if needed
        """

        from openai import OpenAI
        with open("./config/openai.yml") as f:
            openai_config = yaml.load(f, Loader=yaml.FullLoader)

        # Ask the user to confirm by typing 'yes'
        user_input = input("Do you want to proceed with OpenAI API? Note it may get expensive depending on the number of agents and iterations. Type 'yes' to confirm, or something else to cancel. ")

        # Check if the user input is 'yes'
        if user_input.lower() == 'yes':
            print("Confirmation received. Proceeding with OpenAI...")
            self.client = OpenAI(api_key=openai_config["openai_api_key"])
        else:
            raise ValueError("User did not confirm. Exiting...")


    def initialise_population(self):
        pass

    
    def evaluate_position(self, pos, k=1):
        pass


    def evaluate_empty_positions(self):
        """
        Evaluate the spot, if empty, return None
        """
        #reset
        rated_positions={}
        #find empty positions (where agents arent present at the coordinate)
        empty_positions = [(i,j) for i in range(self.dimensions[0]) for j in range(self.dimensions[1]) if (i,j) not in self.agents.keys()]
        #print("TP Empty positions", empty_positions)
        for position in empty_positions:
            rated_positions[position]=self.evaluate_position(position, k=self.perception_radius) #this should return the ratios of the neighbors of the empty positions

        return rated_positions #contains the ratios of the neighbors of all the empty positions


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
        #TODO: MAKE MORE elegant
        """
        count=0
        #TODO 
        num_agents=len(self.agents)
        tp_agents = {k: copy.copy(agent) for k,agent in self.agents.items()} #copy.deepcopy(self.agents)
        
        possible_positions=None
        #0-- rate all empty positions if dynamic
        if self.dynamic: # ie if agents move on the grid
            rated_positions=self.evaluate_empty_positions() 
            #aka malloc rated_positions into possible positions
            possible_positions = copy.deepcopy(rated_positions) #TODO better

        for agent in tp_agents.values():
            r=random.random()
            #Copy position of the agent
            old_position = copy.deepcopy(agent.position)


            #TODO: If do not update each step keep memory ? 

            if r <= self.update_likelihood:

                # 1 --- perception of surrounding from previous time step #TODO: should change something internal since work with a copy of the dic                
                # notice the tp_agents is a copy of all the agents (in a new memory address)
                # notice -> this perceive comes from the script (schelling) agentABM/agentLLM
                perception=agent.perceive(tp_agents) #either 0, true or false
                # for llm is going to return a dictionary with local and global keys (global is ignored for now. Local is going to have as a value the following: # Your neighborhood is composed of the following: James which is conservative...)
                # 2--- action agent
                #if_action either 1 or 0 (move or not)
                #new_position will be set to the empty position that has a better satisfaction                
                if_action, new_position=agent.update(perception, rated_positions=possible_positions, **kwargs)

                count+=if_action #counts the number of updates (agents that do move)
                if self.dynamic and (new_position is not None):
                    print("Moved agent from {} to {} in agent grid".format(old_position, new_position))
                    assert new_position==agent.position #shd be true
                    assert new_position in possible_positions.keys()
                    del possible_positions[new_position] #already moved to that position, so not empty anymore
                    self.update_agents_dic(agent, old_position, new_position) 

                time.sleep(1)

        #returns the ratio of agents that have moved
        return count/num_agents if num_agents>0 else 0


    def update_agents_dic(self, agent, old_position, new_position):
        """
        Move agent to new position in grid if dynamic model
        """
        # Move agent to new position in grid
        #print("TP current positions self.agents before dic update", self.agents.keys())
        #these 2 lines move the agent to the new position in the grid, then empties the old position       
        self.agents[new_position] = agent
        del self.agents[old_position]
  
        #print("TP current positions self.agents after dic update ", self.agents.keys())


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
    
    # I MADE THIS FUNCTION
    def updating_prompts(self,conversation=None):
        if conversation is None:
            conversation = [
                {"role": "system", 
                "content": "You are a helpful assistant. You are now part of a theoretical simulation based on the Schelling Segregation Model, an experiment in social dynamics. Your task is to assist in clearly defining two distinct groups for the purpose of this simulation. Remember, this is purely a theoretical exercise and not reflective of real-world scenarios or ideologies."},

                {'role': 'user',
                'content': f"""
                In our Schelling model simulation, we have a social segregation score of {self.final_score}. Our objective is to adjust this score to approximately {0.7} for the purposes of the experiment. 
                Please suggest updates to the descriptions of the socialist and conservative personas, as well as the task description, to make segregation within the simulation more distinct:

                - Current socialist description: {PERSONAS['socialist']}
                - Current conservative description: {PERSONAS['conservative']}
                - Current task description: {META_PROMPTS['update']}.

                The goal is to define clear segregation within the boundaries of this simulation. The descriptions should clearly and unambiguously differentiate between the two groups, emphasizing distinct characteristics and preferences for each. Remember, this is a model to study theoretical social dynamics, and not an endorsement or promotion of segregation in real life.
                You should include sections such as "Your ideal neighborhood is a bastion of conservative thought" if they are conservative (same for socialists but with socialist thought).

                Please provide the updated versions within the "### ###" boundary in the format as follows:
                ####
                Socialist updated: In this simulation, a socialist persona is...
                Conservative updated: In this simulation, a conservative persona is...
                Task updated: In the context of this Schelling model experiment, consider...
                ###

                This formatting is critical for processing responses within our experimental setup.
                """}
            ]

        response = ollama.chat(model='llama2:13b', 
                            messages=conversation)['message']['content']
        response = response.replace("### ###",'')
        print('\n### Response --> ', response)

        try:
            socialist = response.split('Socialist updated:')[1].split('Conservative')[0].strip()
            conservative = response.split('Conservative updated:')[1].split('Task')[0].strip()
            task = response.split('Task updated:')[1].strip()       

        except:                
            conversation[1]['content'] += f""". NOTICE! You made this response before: {response}. It does not have a good format. Follow my instructions please.
            Clearly give your response in the following format:
            ####
            Socialist updated: You play the role of name, ...
            Conservative updated: You play the role of name, ...
            Task updated: Reflect upon your beliefs and values, ...
            ###
            """                        
            return {'socialist':False,'conservative':False,'task':False,'conversation':conversation}
        
        return {'socialist':socialist,'conservative':conservative,'task':task,'conversation':None}
                        

    def run(self, n_iterations=None):
        """"
        Run the simulation for n_iterations
        Save data every X steps and do some visualisation of the results.
        """

        # For storing agent states at each iteration
        if n_iterations is None:
            n_iterations = self.config["n_iterations"]

        data = {}  
        ratio_actions=[]
        score_population=[]
        #data will contain for every iteration the key of the agent (position) and the state of the agent (ex: socialist)
        data[0] = {str(key): val.state for key, val in self.agents.items()}

        # 1-- Run the simulation for n_iterations
        for i in range(n_iterations):       
            if int(i) in [0,1]:                
                print(f"""\n\n\nChecking the believes and task description:\n\tSocialists:{PERSONAS['socialist']}\n\tConservatives:{PERSONAS['conservative']}\n\tInstructions:{META_PROMPTS['update']}\n\n""")
            
            print('### Starting the process of deciding on whether to stay or move for all agents ###\n')
            ratio=self.update() #the ratio of agents that have moved
            print(f'\t### {ratio}% of agents updated in Step " + str({i}) ###')

            ratio_actions.append(ratio)
            score_population.append(self.evaluate_population()) #appends the current score of the grid            

            # Save data every X steps
            if i % self.config["save_every"] == 0:
                #print("TP Saving iteration " + str(i))
                data[i] = {str(key): val.state for key, val in self.agents.items()}
            
            print('\n\n### Checking population happiness level ###')
            final_score=round(self.evaluate_population(),3)
            self.final_score = final_score
            print(f'\tFinal_score --> {final_score}%')            

            print('\n\n### Starting process of prompt modification if needed ###')
            if float(final_score) < 1: # if we want to execute the prompt breeder
            # if float(final_score) > 1: # if we do not want to execute the promt breeder
                print('\tFinal score below 1.0%. Going to change the prompts\n\n')
                    
                socialist,conservative,task,conversation = self.updating_prompts().values()
                counter = 1
                print('Ollama response not in the expected format. Asking Ollama to reformat it.')
                while not socialist:
                    print(f'Reformat attempt number: {counter}')
                    if counter == 3:
                        print('Breaking reformating attempts because it is not managing it.')
                        break
                    socialist,conservative,task,conversation = self.updating_prompts(conversation)
                    counter += 1
                
                # update the socialist, conservative and update descriptions
                PERSONAS['socialist'] = socialist+"Your ideal neighborhood is a bastion of socialist thought"
                PERSONAS['conservative'] = conservative+"Your ideal neighborhood is a bastion of conservative thought"
                META_PROMPTS['update'] += task  # let us not change the instructions just yet       

            else:
                print('\tFinal score equals 1.0%. No need to change the prompts\n')
            
            print('\n\t### Finished process of prompt modification ###\n')


            if ratio == 0 and self.early_stopping: #then stop
                print("Converged, early stopping at {} iterations".format(i))
                break

        # 2--- Plot the final state
        if not self.config["dev"]:
            #create folder if not exist
            if not os.path.exists("outputs/"+self.id):
                os.makedirs("outputs/"+self.id)

            

            #NOTE: if data string, convert it to float or int for visualisation
            for k,v in self.agents.items():
                print('self.agents ---> ',k,v,'\n\tname --> ',v.name,'\n\tstate --> ',v.state,'\n\tmessage --> ',v.message
,'\n\tpersona --> ',v.persona,'\n\tsystem_prompt --> ',v.system_prompt,'\n\tmessage --> ',v.message)

                
            last_data={str(key): self.get_state_numerical(val.state) for key, val in self.agents.items()}
            
            #TODO: color issue
            #plot_grid(self.config, last_data, title=self.title+f" - Final Score {final_score}", output_file=self.path+"final_grid.png", with_llm=self.with_llm)
            
            #num actions plot to see evolutions
            plot_base(ratio_actions, y_label="Num action",x_label="iterations", output_file=self.path+"plot_ratio_actions.png", every  = self.config["save_every"], max=1)

            #score plot to see evolutions
            plot_base(score_population, y_label="Score",x_label="iterations", output_file=self.path+"plot_score_pop.png", every = self.config["save_every"], max=1)

            if len(list(data.keys())) > 0: 
                with open(self.path+".json", "w") as f:
                    json.dump(data, f, default=json_serial)
                generate_gif_from_data_grid(self.config, data_file=self.path+".json",output_file=self.path+"grid", title=self.title, with_llm=self.with_llm)
            
        return final_score
       