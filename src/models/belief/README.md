# Belief Propagation 

tbd.


## Belief LLM Model

**Initialisation**
Agents are placed on a grid n*n.
Each of them is either a 'believer' or a 'nonbeliever' and initialised with:
- a corresponding **meta-state** (system prompt, representing character ):
    For instance, for a believer:
    ```
    system_prompt = You play the role of #name.
    You inherently distrust authoritative figures, government bodies, and major institutions. You believe these entities often have hidden agendas and might be deceitful.
    You give more weight to personal stories and anecdotes over statistical data or expert opinions. If someone you know experienced it, you believe it must be true.
    ```

- an initial **state** (sets of current thoughts/beliefs), initialised depending on believer/non believer id.
    For instance, for a believer:
    ```
    state = You've recently heard that mainstream sources are not as reliable as they claim. This has made you question their recent reports and statements.
    You give more weight to personal stories and anecdotes over statistical data or expert opinions. If someone you know experienced it, you believe it must be true.
    ```


- an initial **message** to share with its neighbors, initialised depending on believer/non believer id:
    For instance, for a believer:
    ```
    message = A whistleblower from an international health organization has suggested that the real number of coronavirus cases is ten times higher than reported figures due to an alleged global cover-up by major powers to prevent widespread panic.
    ```
    Or:
    ```
    message = Just heard that eating raw garlic daily can prevent you from catching the coronavirus. Better start now!"
    ```
 


**At each steps**:

1. Each agent **perception** is compose of its self perception + local perception of neighbors interactions + global news possibly:
```
perceptionPROMPT = """Here is your current beliefs and thoughts: #state.
    Here are the persona you have recently interacted with: 
    #name, which shared the following: #message
    #name, which shared the following: #message
    ...
    Here is the recent global news you have been exposed to: #global_perception.
    """
```

2. Each agent may **update its internal state** (beliefs).
Here we encourage LLM to first reflect on the context they are in, before giving their final answer.
```
updatePROMPT = """Reflect upon this context, to see if and how #name has evolved its thoughts and beliefs.
You (#name) can decide either to update your beliefs, or to keep your beliefs. Your response should include:
(1) An analysis of the context and cognitive mechanisms at play in #name head recently. 
(2) The decision: answer "[KEEP]" if #name has not modified its beliefs, or "[CHANGE]" followed by the new set of beliefs #name behold."""
```


3. Each agent decide **what message to transmit** to its neigbor at the next step:
```
transmitPROMPT = """ Reflect upon this context, and decide if #name should share something to its neighbors.
You can either decide not to share anything by answering [NONE] or to share something by answering [SHARE] followed by the message you want to share with your community. Do not repeat yourself.
```


## Belief ABM Model
tbd