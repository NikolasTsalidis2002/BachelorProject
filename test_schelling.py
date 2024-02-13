

from src.models.schelling import model as schelling_model
import os
import yaml
from test import test_llm_experiment, test_abm_experiment



############## RUN LLM EXPERIMENT ################
 # 1-- Load config file
config_path = os.path.join(os.getcwd(), "config/schelling.yml")
with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print(f"\nConfig:\n{config}")


# 2-- Change to dummy parameters, for testing
config["n_iterations"] = 2
config["save_every"] = 1
config["grid_size"] = [3, 3]

# 3-- Run experiment with LLM
test_llm_experiment(config, module=schelling_model)



############## RUN ABM  EXPERIMENT ################
# 1-- Load config file
config_path = os.path.join(os.getcwd(), "config/schelling.yml")
with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print(f"\nConfig:\n{config}")


# 2-- Change to dummy parameters, for testing
config["n_iterations"] = 2
config["save_every"] = 1
config["grid_size"] = [3, 3]


# 2-- Run experiment with ABM
test_abm_experiment(config, module=schelling_model)




#TODO: erase simulation