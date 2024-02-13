import matplotlib.pyplot as plt
import itertools
import random
from src.model import GridModel
from src.agent import GridAgent
import numpy as np

# from llama_cpp import Llama #https://llama-cpp-python.readthedocs.io/en/latest/api-reference
from src.visualize import plot_distribution_hist
from src.utils.utils import sample_mixture_gaussian
from src.agentLLM import GridLLMAgent
from src.models.schelling.prompts.meta import META_PROMPTS


class SchellingLLMModel(GridModel):
    def __init__(self, config, id=""):
        param_name = config["focus_parameter_llm"]
        param_value = config["parameters_llm"][param_name]
        path = "outputs/{}/p={}_it={}".format(id, param_value, config["n_iterations"])
        title = f"Schelling LLM, w/ {param_name}={param_value}."
        super().__init__(config, id=id, with_llm=True, title=title, path=path, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"

        # create positions as positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        num_agents = sum(num_agents_by_type)
        self.n_empty = len(self.positions) - sum(num_agents_by_type)
        self.empty_positions = self.positions[: self.n_empty]

        bias = num_agents_by_type[0] / (num_agents_by_type[0] + num_agents_by_type[1])
        self.beliefs = self.initialise_beliefs_population(num_agents_by_type, bias, self.config["parameters_llm"]["polarization"])
        for n in range(num_agents):
            self.agents[self.positions[n]] = SchellingLLMAgent(self.config, state=self.beliefs[n], position=self.positions[n])
        self.personas = self.personas

    def initialise_beliefs_population(self, num_agents_by_type, bias, polarization):
        """
        #TODO: CASE MORE THAN 1 dimension
        #TODO Implement better initialisation; with bias shifting setup rather
        """

        num_agents = sum(num_agents_by_type)
        std = 1.0 - polarization
        mu = 3 * polarization
        beliefs_raw = sample_mixture_gaussian(bias, means=[-mu, mu], std=[std, std], n_samples=num_agents)
        print("Generated Population:", beliefs_raw)
        beliefs = [int(round(b)) for b in beliefs_raw]

        # Clip to ensure beliefs are in the valid range [-3, 3]
        # NOTE: Currently clipped assymetrically and shifted since do not want centrist #TODO: TP
        beliefs = np.clip(beliefs, -2, 3)
        beliefs = [belief - 1 if belief <= 0 else belief for belief in beliefs]

        plot_distribution_hist(
            beliefs,
            mu=[-mu, mu],
            std=[std, std],
            scale=num_agents_by_type,
            xlabel="Political Belief",
            title="Initial Distribution of Political Beliefs",
            path=f"./outputs/{self.id}/distribution_{polarization}.png",
        )

        return beliefs

    def check_similarity_state(self, state1, state2):
        if state1 * state2 > 0 or state1 * state2 == 0:
            return True
        return False

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        """
        # TODO: in case more than 2 types, check ratio of similar types among neighbors

        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        if len(neighbors) == 0:
            return [1 for i in range(len(self.personas))]

        ratios = []
        for i, persona in enumerate(self.personas):
            state = -1 if i == 0 else 1
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, state)])
            ratios.append(float(count_similar) / len(neighbors))

        return ratios

    def evaluate_population(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []
        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
        return sum(similarity) / len(similarity)


class SchellingABMModel(GridModel):
    def __init__(self, config, id=""):

        # ABM model
        param_name = config["focus_parameter_abm"]
        param_value = config["parameters_abm"][param_name]
        param_value = str(int(param_value * 100)) + "%"  # TODO change if change param tested
        path = "outputs/{}/p={}_it={}".format(id, param_value, config["n_iterations"])
        title = f"Schelling ABM, w/ {param_name}={param_value}."
        super().__init__(config, id=id, with_llm=False, title=title, path=path, dynamic=True)

    def initialise_population(self):
        """
        A grid or a neighborhood is initialized with a certain number of agents of different types. Some cells are left empty.
        Agents are randomly placed in this grid, with a certain percentage of the grid left unoccupied.
        """
        assert sum(self.ratio) <= 1, "Sum of ratio must be <=1"

        # create positions as positions in grid
        ranges = [range(dimension) for dimension in self.dimensions]
        self.positions = list(itertools.product(*ranges))
        random.shuffle(self.positions)

        # number positions by types:
        num_agents_by_type = [int(ratio * len(self.positions)) for ratio in self.ratio]
        self.n_empty = len(self.positions) - sum(num_agents_by_type)
        self.empty_positions = self.positions[: self.n_empty]
        count = self.n_empty

        # Initialise belieds population
        for i in range(len(self.personas)):
            for j in range(num_agents_by_type[i]):
                self.agents[self.positions[count]] = SchellingABMAgent(self.config, state=i, position=self.positions[count])
                count += 1

    def evaluate_position(self, pos, k=1):
        """
        Evaluate position according to ratio of similar types among neighbors
        #TODO: in case more than 2 types, check ratio of similar types among neighbors

        """
        neighbors = [
            self.agents[(x, y)]
            for x in range(pos[0] - k, pos[0] + k + 1)
            for y in range(pos[1] - k, pos[1] + k + 1)
            if (x, y) in self.agents.keys()
        ]
        if len(neighbors) == 0:
            return [1 for i in range(len(self.personas))]
        count_similar = [sum([1 for n in neighbors if n.state == i]) for i in range(len(self.personas))]
        ratios = [float(count_similar[i]) / len(neighbors) for i in range(len(self.personas))]
        return ratios

    def evaluate_population(self):
        """
        Mean of similarity score, where count the ratio of similar types among neighbors
        """
        similarity = []
        for agent in self.agents.values():
            neighbors = agent.get_neighbors(self.agents, k=self.perception_radius)
            count_similar = sum([1 for n in neighbors if self.check_similarity_state(n.state, agent.state)])
            try:
                similarity.append(float(count_similar) / (len(neighbors)))
            except:
                similarity.append(1)
        return sum(similarity) / len(similarity)


class SchellingABMAgent(GridAgent):

    def __init__(self, config, position=None, state=None):
        """
        Note here the state of the Schelling agent is is type .
        """
        super().__init__(config, position=position, state=state)

        for key, val in config["parameters_abm"].items():
            setattr(self, key, val)

    def check_similarity_state(self, state1, state2):
        return state1 == state2

    def perceive(self, agents):
        """
        Return the satisfaction score from its neighbors

        """

        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])

        # if no neighbors, agent is happy and do not move ? #TODO
        if len(neighbors) == 0:
            return 0

        # Check ratio of similar types among neighbors if below threshold, unsatisfied
        count_similar = sum([1 if n.state == self.state else 0 for n in neighbors])
        self.score = float(count_similar / len(neighbors))
        unsatisfied = self.score < self.similarity_threshold

        return unsatisfied

    def update(self, perception, rated_positions=None):
        """
        Move the agent to a new position if unsatisfied, based on n-dimensional space.
        """
        if perception == 0:
            return 0, None

        desirable_positions = {k: v for k, v in rated_positions.items() if v[self.state] > self.score}

        # If no desirable positions available
        if not desirable_positions:
            return 0, None

        # Choose a new position based on desirability weights
        weights = [v[self.state] for v in desirable_positions.values()]
        positions = list(desirable_positions.keys())
        new_index = np.random.choice(len(positions), p=np.array(weights) / sum(weights))
        new_position = positions[new_index]

        # Update the agent's position to the new selected position in n-dimensional space
        self.position = new_position

        return 1, new_position


class SchellingLLMAgent(GridLLMAgent):
    # TODO DOUBLE INHERITANCE LLM AGENT AND GRID AGENT...

    def __init__(self, config, position=None, state=None):
        """
        Note here the state of the Schelling agent is is type .
        """

        persona = "socialist" if state < 0 else "conservative"
        super().__init__(config, position=position, state=state, persona=persona)
        self.message = self.get_state_as_text()

        # setup meta prompts
        self.PROMPTS = META_PROMPTS

    def get_state_as_text(self):
        """
        Return the textual state of the agent
        """
        return self.persona

    def check_similarity_state(self, state1, state2):

        if state1 < 0 and state2 <= 0:
            return True
        elif state1 > 0 and state2 >= 0:
            return True
        elif state1 == 0 and state2 == 0:
            return True
        else:
            return False

    def perceive(self, agents, global_perception=None):
        """
        Perception by default is made by one own state, some global perception and some local perceptions (neighbor messages)
        """

        prompts = self.PROMPTS["perception"]

        perception = {}

        perception["self"] = prompts["self"].format(name=self.name, state=self.get_state_as_text())

        neighbors = self.get_neighbors(agents, k=self.config["parameters"]["perception_radius"])

        perception["local"] = ""
        if len(neighbors) > 0:
            shared = ""
            for n in neighbors:
                if n.message is not None and n.message != "":
                    shared += prompts["local_neighbors"].format(name=n.name, message=n.message)
            if shared != "":
                perception["local"] = prompts["local"].format(local_perception=shared) + shared

        perception["global"] = ""
        if (global_perception is not None) and "global" in prompts.keys():
            perception["global"] = prompts["global"].format(global_perception=global_perception)

        # Update score perception
        self.score = (
            1 if len(neighbors) == 0 else sum([1 for n in neighbors if self.check_similarity_state(n.state, self.state)]) / len(neighbors)
        )

        return perception

    def update(self, perception, rated_positions=None):
        """
        Move house if unsatisfied  #TODO: could also change type? Being influenced?

        Args:
            perception: here num_neighbors_by_type
        """
        context = self.get_context_from_perception(perception)
        type = 0 if self.state < 0 else 1  # TODO issue with centrist

        # If satisfied, do nothing
        if perception is None:
            return 0, None

        # filter dictionary to only keep better positions #TODO:
        desirable_positions = {k: v for k, v in rated_positions.items() if v[type] > self.score}

        # if no empty home
        if len(list(desirable_positions.keys())) == 0:
            return 0, None

        # Check if want to move
        prompt = context + self.PROMPTS["update"].format(name=self.name)
        response = self.ask_llm(prompt, max_tokens=10)
        if "STAY" in response:
            print(f"TEMP DEBUG: STAY agent belief {self.state} and context {context}")
            return 0, None

        # TODO: make choose where stay ?
        print(f"TEMP DEBUG: MOVE agent belief {self.state} and context {context}")

        # If unsatisfied, move to empty house
        # sample from rated positions keys with weights the positions rat
        weights = [v[type] for v in desirable_positions.values()]  # similar ratio to sample neighborhood
        positions = list(desirable_positions.keys())
        new_index = np.random.choice(len(positions), p=np.array(weights) / sum(weights))
        new_position = positions[new_index]
        self.x = new_position[0]
        self.y = new_position[1]

        return 1, new_position

    def transmit(self, perception):
        """
        Here message transmitted is its own state (here expressed as belief, with text)
        """
        self.message = self.get_state_as_text()
