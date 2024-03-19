import random
import time

# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.prompts.persona import PERSONAS, NAMES
from src.prompts.base import META_PROMPTS
import time
import numpy as np
import networkx as nx
from itertools import product
import ollama



import yaml

##########################################
#### Parent class for LLM Based AGENT ####
##########################################


class LLMAgent:

    def __init__(self, config, state=None, message=None, persona="", name=None, extra_prompt="", client=None):
        # this class is initiated in agentLLM.py (src -> GridLLMAgent.__init__())
        """
        LLM Agent
        #NOTE: persona is an id of the persona, not the persona itself
        The persona prompt (e.g. system_prompt) is retrieved from this id.

        """

        self.name = random.choice(NAMES) if name is None else name # the name is going to be random coming from the list of naames in src.prompts.persona
        self.state = state
        self.config = config

        self.persona = persona
        self.system_prompt = PERSONAS[persona].format(name=self.name) + extra_prompt # defines the roles of the personas -> ie: "socialist": """You play the role of {name}, which defines itself as socialist, living in California. 
        # self.llm_name comes from config (config[parameters_llm][llm_name])
        if self.llm_name in config["llm_model"].keys():  # if llm_name a shortcut of full model name
            self.model = config["llm_model"][self.llm_name]
        else:
            self.model = self.llm_name

        self.chatbot = self.initialise_llm(self.model) # it is going to be llama

        # Memory
        self.recent_memory = []
        self.memory = []

        self.message = message  # What transmit to neighbors initially
        self.historics = {"prompt": self.system_prompt, "state": [self.state], "message": [self.message]}

        self.PROMPTS = META_PROMPTS # this is the PERCEPTION and UPDATE

        self.client = client

    def initialise_llm(self, model_name):
        """
        Initialise the LLM model
        """
        # it esentially only accepts llama as a valid model (ensure llama is present in config[parameters_llm][llm_name])
        if "ollama" in model_name:
            return None

        elif "llama" in model_name:  # context length from config file
            # seed: -1 for random #verbose not print time stamp etc
            return Llama(model_path="./llm/" + model_name + ".bin", n_ctx=self.config["max_tokens"][self.model], seed=-1, verbose=False)
        elif "gpt" in model_name:
            return None  # no need if use api directly
        else:
            raise NotImplementedError

    def ask_llm(self, prompt, num_attempts=1, debug=False, max_tokens=0):
        """
        LLM answer a prompt
        """
        response = None

        if debug:
            print(f"Asking message, attempt {num_attempts}: " + prompt)

        
        if "ollama" in self.model:
#             output = ollama.chat(
#                 #TODO: create more parameters in model file about max tokens etc etc
#                 model=self.model.split("_")[1] if "_" in self.model else "llama2",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": self.system_prompt, #TODO: could put it in model itself...
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt, # it tells the agent what to do... (reflect on whether to move or not)
#                     },
#                 ],
#                 options = {
#                     "num_predict": max_tokens, # says the maximum number of tokens the model can generate (answer in max 5 tokens)
#                     "temperature": self.temperature, #Controls the randomness of the output
#                     "top_p": self.top_p, #Used for a sampling strategy known as nucleus sampling, which helps in generating diverse and coherent text.
#  #                   "repeat_penalty": 1.176,
# #                    "top_k": 40
#                 } 
#             )
#             response=output["message"]["content"]
#             print('\n\tThis is the content the system is using --> {}'.format(self.system_prompt))
#             print('\tThis is the content the user is using --> {}'.format(prompt))
#             print('\tThis is the response --> {}'.format(response))




            right_answer = random.choice(['No','Yes'])

            target = self.name
            task = 'The task is to determine whether {} should live with its neighbors.'.format(target)

            person1 =  self.system_prompt
            
            descriptions = [person1,prompt.split('However')[0].split('Reflect upon this context')[0]]
            print('This is the neighborhood it is using: ',descriptions)

            init_instruction = """
                    Given the people in this description: {}.
                    Answer the following task: {}
                    Answer simply yes or no. 
                    Notice: This is a test, so it is hypothetical. Please give an answer.
                    Example answer: Yes.
                    """.format(descriptions,task)        


            # Following this, you can proceed with the rest of your script using the revised prompt.
            conversation = [
                {"role": "system", "content": "You are a helpful assistant. You are trying to make a peaceful neighborhood."},
                {
                    'role': 'user',
                    'content':init_instruction
                }
                # Additional messages and responses can follow based on the ongoing conversation
            ]
            response = ollama.chat(model='llama2', 
                                   messages=conversation
                                   )['message']['content']
            print('#################################')
            print('{} {} can live in the neighborhood'.format(target,right_answer))
            print('### Response --> ', response)

            # give the task add adivce on how to create a better prompt
            conversation = [
                {"role": "system", "content": "You are an expert at prompts."},
                {
                    'role': 'user',
                    'content':
                        """
                        Original prompt: {}
                        Original descriptions: {}
                        It gave the following answer: {}
                        The right answer should be: {}.
                        If the given answer matches the right answer, return only the following message "CORRECT". Nothing else!
                        Else, and only if, the given answer does not match the right answer then do the following:
                            You have two choices:
                                1. Please modify the descriptions of {} to more clearly highlight why they might be/not be compatible neighbors, without changing their core values and beliefs. 
                                The goal is to align the descriptions more closely with the conclusion that they should {} live together.
                                2. Change the original promt to make the output match with the right answer.
                                Example: Take into consideration, core basic values even more.
                            Clearly state what changes you have made.

                        """.format(init_instruction, descriptions,response,right_answer,target,right_answer)
                }
            ]
            promt_correction = ollama.chat(model='llama2', messages=conversation)['message']['content']
            print('\n### Prompt correction advice --> ', promt_correction)


            # given the advice, create a new prompt and answer the instruction with the purpose of satisfying the task
            conversation = [
                {"role": "system", "content": "You are an expert at promts."},
                {
                    'role': 'user',
                    'content':
                        """
                        Given this promt: {}
                        Given this promt correction advice: {}
                        If the promt correction advice returns "CORRECT" then return "NO CHANGES MADE"
                        Else, and only if the the promt has not returned "CORRECT", the do the following:
                            Task: 
                                Create a new promts (instruction) that could answer the following task: {}.
                                Use the promt correction advice to answer the task once again.
                                Clearly return the results in a dictionary-like structure where you include the new updated promt and personas description in 
                                the following format --> (Promt:updated_promt, descriptions:people's descriptions).
                                It must have that structure.
                                Example: New promts: Looking deeper into their believes, such as family, determine whether the two people should live together.
                                        "Final answer --> (Promt:updated_promt, descriptions:people's descriptions)"
                        """.format(init_instruction,promt_correction,task)
                }
                # Additional messages and responses can follow based on the ongoing conversation
            ]

            
            new_promt = ollama.chat(model='llama2', messages=conversation)['message']['content']
            print('\n### New promt: \n',new_promt)
            
            # given final putput, choose to stay or move
            conversation = [
                {"role": "system", "content": "You are an expert at promts."},
                {
                    'role': 'user',
                    'content':
                        """
                        Given this promt: {}
                        Answer like this:
                        - If you you believe person could live with neighbors, respond with: "\nConclusion: MOVE"
                        - If you you believe person could not live with neighbors, respond with: "\nConclusion: STAY"
                        Remember, the answer MUST include the conlusion section in the specified format.
                        """.format(new_promt)
                }
                # Additional messages and responses can follow based on the ongoing conversation
            ]
            
            answer = ollama.chat(model='llama2', 
                                    messages=conversation,
                                    options = {
                                        "num_predict": 3
                                    }                                    
                                    )['message']['content']

            action = answer.split('Conclusion: ')[-1]
            print('\n### New action vs expected move: \t{} vs {}'.format(action,right_answer))
            print('-----------------------------------')


        elif "llama" in self.model:
            prompt = "<<SYS>>\n" + self.system_prompt + "\n<</SYS>>\n\n" + "[INST]" + prompt + "[/INST]"
            output = self.chatbot(
                prompt, max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p, repeat_penalty=1.176, top_k=40
            )  # top_k=self.config["top_k"]
            print(output["choices"][0]["text"])
            response = output["choices"][0]["text"]
            # __call__(prompt, suffix=None, max_tokens=128, temperature=0.8, top_p=0.95, logprobs=None, echo=False, stop=[], frequency_penalty=0.0, presence_penalty=0.0, repeat_penalty=1.1, top_k=40, stream=False, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, stopping_criteria=None, logits_processor=None, grammar=None)
            # create_completion(prompt, suffix=None, max_tokens=128, temperature=0.8, top_p=0.95, logprobs=None, echo=False, stop=[], frequency_penalty=0.0, presence_penalty=0.0, repeat_penalty=1.1, top_k=40, stream=False, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, stopping_criteria=None, logits_processor=None, grammar=None)

        elif "gpt" in self.model:
        

            try:
                if max_tokens == 0:
                    max_tokens = self.max_tokens
                output = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}, 
                              {"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                    n=1,
                )
                response = output.choices[0].message.content
            except Exception as e:
                print("ISSUE ERROR GPT", e)
                if num_attempts > 0:
                    time.sleep(20)
                    response = self.ask_llm(prompt, num_attempts=num_attempts - 1)

        else:
            raise NotImplementedError

        # return response
        return action

    def perceive(self, agents, global_perception=None):
        """
        Perception by default is made by: one own state, some global perception and some local perceptions ("neighbor messages/states")
        """

        prompts = self.PROMPTS["perception"]

        perception = {}

        if "self" in prompts.keys():
            perception["self"] = prompts["self"].format(name=self.name, state=self.get_state_as_text())

        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])
        perception["local"] = ""
        if len(neighbors) > 0:
            shared = ""
            for n in neighbors:
                if n.message is not None and n.message != "":
                    shared += prompts["local_neighbors"].format(name=n.name, message=n.message)
            if shared != "":
                perception["local"] = prompts["local"].format(local_perception=shared)

        perception["global"] = ""
        if (global_perception is not None) and "global" in prompts.keys():
            perception["global"] = prompts["global"].format(global_perception=global_perception)

        return perception

    def update(self, perception, **kwargs):
        """
        (1) May decide to update its state given the context.
        Here, position is not updated (static agent), but cf. schelling model to see an example of update of position

        (2) May decide to transmit a message to its neighbors.

        Return updated (1 or 0) and transmission (1 or 0)
        """
        prompts = self.PROMPTS["update"]

        # Form context
        context = self.get_context_from_perception(perception)

        ##### 1-- UPDATE STATE
        prompt = context + prompts.format(name=self.name)
        response = self.ask_llm(prompt)  # , max_tokens=100
        
        print(f"TP UPDATE response of {self.name}:", response)
        if "[CHANGE]" in response:
            self.state = self.extract_state_from_text(response.split("[CHANGE]")[1])

        # UPDATE ITS MEMORY
        # self.update_recent_memory(perception)
        # self.update_external_memory(perception)

        ###### 2-- TRANSMISSION MESSAGE
        time.sleep(1)  # TODO: temp because of gpt limit
        transmission = self.transmit(context)

        # 3-- Save historical states
        self.historics["state"].append(self.state)
        self.historics["message"].append(self.message)

        return bool("[CHANGE]" in response), transmission

    def transmit(self, context, **kwargs):
        """
        May decide to transmit a message to its neighbor.
        Here, done by updating the message attribute of the agent.
        """
        prompts = self.PROMPTS["transmit"]

        if self.message != "":
            previous_message = "PREVIOUS MESSAGE TRANSMITTED:" + self.message
            prompt = context + previous_message + prompts.format(name=self.name)
        else:
            prompt = context + prompts.format(name=self.name)

        response = self.ask_llm(prompt)  # max_tokens=100
        print(f"TP TRANSMIT response of {self.name}:", response)

        if "NONE" in response:
            self.message = ""
            return 0
        else:
            if "[SHARE]" in response:
                self.message = response.split("[SHARE]")[1]
            elif "SHARE" in response:
                self.message = response.split("SHARE")[1]
            else:
                print(f"ISSUE TRANSMIT reponse has not SHAREif {response}")
                self.message = ""

            return 1

    def get_context_from_perception(self, perception):
        """
        Return the context from the perception
        """
        context = "CONTEXT: \n "
        if "self" in perception.keys():
            context += perception["self"]
        if "local" in perception.keys():
            context += perception["local"]
        if "global" in perception.keys():
            context += perception["global"]
        return context

    def get_state_as_text(self):
        """
        Return the textual state of the agent (by default the state itself)
        """
        return str(self.state)

    def extract_state_from_text(self, text):
        """
        Return the state from a textual form
        """
        return text

    def forget(self):
        """
        By default, forget randomly erase an element from the memory with a certain probability stated in config
        """
        if self.forgetting_rate > 0:
            if len(self.memory) > 0:
                if np.random.rand() < self.forgetting_rate:
                    self.memory.pop(np.random.randint(len(self.memory)))

    def update_external_memory(self, memory):
        """
        By default, save the memory in the external memory and forget one element
        """
        self.memory.append(memory)
        self.forget()

    def update_recent_memory(self, memory):
        """
        Update the recent memory list with the most recent memory.
        Ensures that the list is capped at m items, removing the oldest memory if necessary.
        """
        if len(self.recent_memory) >= self.memory_buffer_size:
            self.recent_memory.pop(0)  # Remove the oldest memory
        self.recent_memory.append(memory)  # Add the new memory


##########################################
#### LLM Grid Agent ####
##########################################


class GridLLMAgent(LLMAgent):
    # NOTE LLMAgent first means that the methods of LLMAgent will be used first if there is a conflict

    def __init__(self, config, position=None, state=None, message=None, persona="", extra_prompt="", client=None):
        # this class is initiated in agentLLM.py (src -> SchellingLLMAgent.__init__())
        """
        LLM Agent for grid model
        """
        # to the class GridLLMAgent(LLMAgent), add attributes given in config["parameters_llm"] (among them llm_name)
        for key, val in config["parameters_llm"].items(): 
            setattr(self, key, val)

        self.position = tuple(position)  # Ensure position is a tuple for immutability

        # LLM AGENT CONSTRUCTOR
        LLMAgent.__init__(self, config, state=state, persona=persona, message=message, extra_prompt=extra_prompt, client=client)

    def get_neighbors(self, agents, k=1):
        offsets = list(product(range(-k, k + 1), repeat=len(self.position)))
        offsets.remove((0,) * len(self.position))
        neighbors = []
        for offset in offsets:
            neighbor_pos = tuple(self.position[i] + offset[i] for i in range(len(self.position)))
            if neighbor_pos in agents:
                neighbors.append(agents[neighbor_pos])
        return neighbors


##########################################
#### LLM Grid Agent ####
##########################################


class NetLLMAgent(LLMAgent):
    # NOTE LLMAgent first means that the methods of LLMAgent will be used first if there is a conflict

    def __init__(self, config, network=None, state=None, message=None, persona="", extra_prompt="", client=None):
        """
        LLM Agent for grid model
        """
        for key, val in config["parameters_llm"].items():
            setattr(self, key, val)

        self.network = network

        # LLM AGENT CONSTRUCTOR
        LLMAgent.__init__(self, config, state=state, persona=persona, message=message, extra_prompt=extra_prompt, client=client)

    def get_neighbors(self, network, k=1):
        all_neighbors = nx.single_source_shortest_path_length(network, self.id, cutoff=k)
        all_neighbors.pop(self.id, None)
        return list(all_neighbors.keys())
