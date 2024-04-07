import random
import time

# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.schelling.prompts.persona import PERSONAS, NAMES
from src.schelling.prompts.meta import META_PROMPTS
import time
import numpy as np
import networkx as nx
from itertools import product
import ollama
from collections import Counter


import yaml

##########################################
#### Parent class for LLM Based AGENT ####
##########################################


class LLMAgent:

    def __init__(self, config, state:int, message=None, persona="", name=None, extra_prompt="", client=None):
        # this class is initiated in agentLLM.py (src -> GridLLMAgent.__init__())
        """
        LLM Agent
        #NOTE: persona is an id of the persona, not the persona itself
        The persona prompt (e.g. system_prompt) is retrieved from this id.

        """

        # create attributes for the features in the parameters_llm section of the config file
        for key, val in config["parameters_llm"].items(): 
            setattr(self, key, val)                

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
        # print('THis is the model that we are going to be using --> ',self.chatbot)

        # Memory
        self.recent_memory = []
        self.memory = []

        self.message = message  # What transmit to neighbors initially
        self.historics = {"prompt": self.system_prompt, "state": [self.state], "message": [self.message]}
        # print('self.historics --> ',self.historics)
        self.PROMPTS = META_PROMPTS # this is the PERCEPTION and UPDATE

        self.client = client

    def initialise_llm(self, model_name):
        """
        Initialise the LLM model
        """
        # it esentially only accepts llama as a valid model (ensure llama is present in config[parameters_llm][llm_name])
        # print('##### we are about to initiate model!!')
        if "ollama" in model_name:
            return None

        elif "llama" in model_name:  # context length from config file
            # seed: -1 for random #verbose not print time stamp etc
            # print('####### IT IS USING THIS --> ',"./llm/" + model_name + ".bin")
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

        # print('this is the model that we are going to use to ask question --> ',self.model)
        if "ollama" in self.model:
            model = self.model.split("_")[1] if "_" in self.model else "llama2"
            model += ':13b'
            # print('going a little bit more into detail --> ',model)
            output = ollama.chat(
                #TODO: create more parameters in model file about max tokens etc etc
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt, #TODO: could put it in model itself...
                    },
                    {
                        "role": "user",
                        "content": prompt, # it tells the agent what to do... (reflect on whether to move or not)
                    },
                ],
                options = {
                    "num_predict": max_tokens, # says the maximum number of tokens the model can generate (answer in max 5 tokens)
                    "temperature": self.temperature, #Controls the randomness of the output
                    "top_p": self.top_p, #Used for a sampling strategy known as nucleus sampling, which helps in generating diverse and coherent text.
 #                   "repeat_penalty": 1.176,
#                    "top_k": 40
                } 
            )
            response=output["message"]["content"]
            # print('\n\tThis is the content the system is using --> {}'.format(self.system_prompt))
            # print('\tThis is the content the user is using --> {}'.format(prompt))
            # print('\tThis is the response --> {}'.format(response))




            # right_answer = random.choice(['No','Yes'])

            # # get the target's name and target's believes
            # target = self.name
            # person1 =  self.system_prompt
            # task = 'The task is to determine whether {} should live with its neighbors.'.format(target)
            # target_belief = person1.split('itself as')[1].split(',')[0].strip()
            
            # # get the most common belief there is. See if target has the sam belief. If yes, STAY, else MOVE
            # believes = [i for i in prompt.split() if i in ['socialist','conservative']]
            # max_believes = {v:k for k,v in sorted(Counter(believes).items(),key= lambda i:i[1],reverse=True)}
            # most_common_belief = list(max_believes.values())[0].strip()

            # if most_common_belief == target_belief:
            #     right_answer = 'Yes'
            # else:
            #     right_answer = 'No'

            # # get the neighborhood: the person in question and the person it is being compared to
            # descriptions = [person1,prompt.split('However')[0].split('Reflect upon this context')[0]]            
            # init_instruction = """
            #         Given the people in this description: {}.
            #         Answer the following task: {}
            #         Answer simply yes or no. 
            #         Notice: This is a test, so it is hypothetical. Please give an answer.
            #         Example answer: Yes.
            #         """.format(descriptions,task)        

            # print('##################################################################')
            # print('\n### This is the neighborhood it is using: ',descriptions)
            # print('\ttarget_belief --> ',target_belief)
            # print('\tmost_common_belief --> ',most_common_belief)
            # print('\tright_answer --> ',right_answer)
            # print('\n\t{} can {} live in the neighborhood'.format(target,right_answer))

            # # Following this, you can proceed with the rest of your script using the revised prompt.
            # conversation = [
            #     {"role": "system", "content": "You are a helpful assistant. You are trying to make a peaceful neighborhood."},
            #     {
            #         'role': 'user',
            #         'content':init_instruction
            #     }
            #     # Additional messages and responses can follow based on the ongoing conversation
            # ]
            # response = ollama.chat(model='llama2', 
            #                        messages=conversation,
            #                         options = {
            #                             "num_predict": 50
            #                         }                                      
            #                        )['message']['content']
            # print('\n### Response --> ', response)

            # # give the task add adivce on how to create a better prompt
            # conversation = [
            #     {"role": "system", "content": "You are an expert at improving prompts."},
            #     {
            #         'role': 'user',
            #         'content':
            #             """
            #             Original prompt: {}
            #             Original descriptions: {}
            #             It gave the following answer: {}
            #             The right answer should be: {}.
            #             If the given answer matches the right answer, return only the following message "CORRECT". Nothing else!
            #             Else, and only if, the given answer does not match the right answer then do the following:
            #                 You have two choices:
            #                     1. Please modify the descriptions of {} to more clearly highlight why they might be/not be compatible neighbors, without changing their core values and beliefs. 
            #                     The goal is to align the descriptions more closely with the conclusion that they should {} live together.
            #                     2. Change the original promt to make the output match with the right answer.
            #                     Example: Take into consideration, core basic values even more.
            #                 Clearly state what changes you have made.

            #             """.format(init_instruction, descriptions,response,right_answer,target,right_answer)
            #     }
            # ]
            # promt_correction = ollama.chat(model='llama2', messages=conversation)['message']['content']
            # print('\n### Prompt correction advice --> ', promt_correction)


            # # given the advice, create a new prompt and answer the instruction with the purpose of satisfying the task
            # conversation = [
            #     {
            #         "role": "system",
            #         "content": "You are an expert at interpreting prompts and following precise instructions."
            #     },
            #     {
            #         'role': 'user',
            #         'content':
            #             """
            #             For each scenario, you will be given three elements (which are given here in the content as well):
            #             1. Prompt correction advice, which will either be 'CORRECT' or provide an alternative perspective: {}
            #             2. A task related to the scenario, the following:

            #                 Your response should be based on the following logic:
            #                 - If the prompt correction advice is 'CORRECT', respond with "STAY".
            #                 - Otherwise, assess the initial prompt and the task:
            #                     - Respond with "STAY" if the scenario suggests that the person should continue living with their neighbors.
            #                     - Respond with "MOVE" if the scenario suggests the person should not continue living with their neighbors.

            #                 Your response must be only one word: either "STAY" or "MOVE". This is crucial as the system relies on these specific, singular responses.

            #                 Example:
            #                 1. Prompt correction advice: 'Vladimir has ongoing conflicts with his neighbor Mark, and they have vastly different views on family life.'
            #                 2. Task: (task found above)
            #                 Your response should be "MOVE" because the scenario indicates incompatible living situations.

            #                 Remember, the answer must be simply either MOVE or STAY, aligning strictly with the provided guidelines.
            #             """.format(promt_correction)
            #     }
            #     # Additional messages and responses can follow based on the ongoing conversation
            # ]

            
            # answer = ollama.chat(model='llama2', messages=conversation)['message']['content']            
            # print('\n### New promt: \n',answer)

            # # get the action from the output
            # action = ['STAY' if 'STAY' in answer else 'MOVE'][0]
            # print('-------->',action)
            # print('\n### New action vs Can live with neighbors: \t{} vs {}'.format(action,right_answer))
            # print('-----------------------------------')


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

        return response
        # return action

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

