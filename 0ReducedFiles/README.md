# :lizard: :space_invader: LLM CSS :space_invader: :lizard: 


## Overview

### Installation
First, install the required packages:
```
pip3 install -r requirements.txt
```

### Quick Run 
The main option for LLM is to rely currently on , which enable to switch between models. https://ollama.com/library
If you want to use OpenAI API for the LLM (not advised, because of costs) you need to first have your OPENAI key in a file called openai.yml in config folder/ with as field openai_api_key: <YOURKEY>
Alternatively, you can download a model in the folder llm/ and use it by specifying in the config file the model_name in parameters_llm
and indicate the full name of the file in llm_model.

Currently two main models are implemented schelling (cf. models/schelling/, both LLM or ABM) and a belief/opinion propagation model (cf. models/belief/, only LLM for now)

To run an experiment, you can simply run:
```
python3 main.py -xp <MODEL_NAME> -a <ABM or LLM>
```

For instance, for LLM Schelling model: (by default LLM models are selected)
```
python3 main.py -xp schelling
```

and for ABM Schelling model:
```
python3 main.py -xp schelling -a abm
```




### Structure
* **Folders**
    - **src/**: main classes needed (cf. below)
        - **models/**: here lies the actual models being implemented upon the library classes
        For now, two examples here: a belief propagation model and a schelling model can be taken as an example and starting point.
        - **llm/**: folder to save the llm models you want to use (if not via api)
        - **config/**

* **Classes**
    - **LLM-Agents**, in agentLLM.py
    LLMAgent, and GridLLMAgent for LLM-Agent positioned on a grid
     - **ABM-agents** (base agent class), in agentABM.py
    GridABMAgent, for agent positioned on a grid
    - **Models**, in model.py
    GridModel, for Model with Grid Substrate
    #TODO: Network based model


### :vulcan_salute: Create your own model :vulcan_salute:
To create a new model, you need to:
1. give a **name** to your model, e.g. 'dummy'
2. create a **folder** in models/ (e.g. models/dummy) with some files called **model.py**, **agentLLM.py**, and/or **agentABM.py**
This file should hold a class called DummyModel and possibly agents classes, for instance DummyLLMAgent or DummyABMAgent. You can take inspiration from existing models and build upon other parent classes.
You may use a prompt folder with the needed prompts, or need specific visualisation functions etc.
3. Create a **config file** with the same name, e.g. dummy.yml that you place in the config/ folder
4. in **main.py**, add some lines at the end of the file to be able to launch your model, looking like:
```
elif xp=="dummy":
    from src.models.dummy import model as dummy_model
    run_experiment(config, module=dummy_model)
```


5. (optional) ideally, create a README.md detailing your model.

### Adding a LLM 
- Either use Ollama: in the config file, you should write ollama_model_name, and ensure it is supported by ollama.
You can also through ollama use a [modelfile] (https://github.com/ollama/ollama/blob/main/docs/modelfile.md)
- or Specify the "llm_name" in the config file, Download your model in the folder src/llm/ 
and Look at the LLM Agent and you MAY need to modify the methods initialise_llm() and ask_llm_() to ensure to fit your model


## Agents in NCA-LLM
By default, LLM-Agents have:
- a **meta-state**, called 'system-prompt' (gpt-terminology), which is the one given each time in preamble before asking them how they act (instantiated from a 'persona' identifier), by defaul fixed (although could be updated too)
- a **state** (string for instance)
- a **message** (string), which is updated and shared with their neighbors.
By default, the agents only access this message, not the full state of their neighbors (form of indirect observability).
- some **neighbors** (if grid substrate or graph: any agent at distance k, k>=1)
#TODO soon:  may instantiate specific **relations state** between agents


By default, at each turn, with a certain likelihood, LLM-Agents do:
1. **perceive** their local environment --i.e. gather the messages shared by their neighbors-- possibly with some extra global information (e.g. news)
2. **update** their state, given this context (i.e. given perception)
3. **transmit** a message to their neighbor (by default same to every neighbors), given this context


## Models

### Schelling
cf. [Schelling Model](https://github.com/claireaoi/nca-llm/tree/main/src/models/schelling)
[cf. README](https://github.com/claireaoi/nca-llm/tree/main/src/models/schelling/README.md)


![Schelling ABM Model](./img/schelling.gif)

### Belief Propagation
cf. [Belief Propagation Model](https://github.com/claireaoi/nca-llm/tree/main/src/models/belief)
[cf. README](https://github.com/claireaoi/nca-llm/tree/main/src/models/belief/README.md)



