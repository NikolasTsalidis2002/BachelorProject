
import agentpy as ap
import numpy as np
import random
import utils
import agentpy as ap
import numpy as np
from consumer import Consumer
from company import EnergyCompany
from producer import EnergyProducer
from state import State
from utils import is_renewable, get_adjacency_matrix, sample_from_beta_distribution, extract_meso,  extract_data, load_config
import networkx as nx
import datetime
import os
import yaml
from visualize import animate, plot_energy_data
import random


class SchellingModel(ap.Model):

    def setup(self, agents_policy_weights=None):
        """Initialise a new list of agents."""
        self.t=0

        if self.p.learning:
            print("RL Agents for Producers")

        # Create agents
        self.consumers = ap.AgentList(self, self.p.n_consumers, Consumer)
        self.energy_companies = ap.AgentList(self, self.p.n_companies, EnergyCompany)
        self.energy_companies[0].sustainability = 1 #prioristy sustainable company #TODO: CHANGE !
        self.energy_companies[1].sustainability = 0
        self.energy_producers = ap.AgentList(self, self.p.n_producers, EnergyProducer)
        self.state = ap.AgentList(self, self.p.n_states, State)

        for agent in self.consumers:
            agent.pick_energy_company(self.energy_companies)

        #all agents together #TODO: ORDER MATTER ...
        self.agents = self.state+self.energy_producers+self.energy_companies + self.consumers

        full_adjacency=get_adjacency_matrix(self.consumers,self.energy_companies,self.energy_producers,self.state)
        # convert numpy array to networkx graph
        graph = nx.from_numpy_array(full_adjacency)
        self.network = self.agents.network = ap.Network(self, graph)  # Optimise: why each agent has a network attribute?
        self.network.add_agents(self.agents, self.network.nodes)

         #SET POLICY WEIGHTS FOR PRODUCERS IF LEARN
        if agents_policy_weights is not None and self.p["learning"]:
            if True: #self.p["populationMARL_perAgent"]:
                assert len(agents_policy_weights) == self.p.nb_learning_agents
                j=0
                for agent in self.agents:
                    if agent.type=="producer":
                        policy_weights=agents_policy_weights[j]
                        agent.set_policy(policy_weights)
                        j+=1



    def update(self):
        """ Record variables after setup and each step. """
        # Record size of the biggest cluster
        # clusters = nx.connected_components(self.buttons.graph)
        # max_cluster_size = max([len(g) for g in clusters]) / self.p.n
        # self.record('ideology', np.array(self.agents.ideology)) 

        #Record data 
        for energy in self.p.energy:
            costs=[prod.data["cost"][energy][-1]/prod.data["produced"][energy][-1] for prod in self.energy_producers if prod.data["produced"][energy][-1]>0]
            self.record('energy_cost'+"_"+energy, np.mean(costs) if len(costs)>0 else 0)

        self.record_data(self.energy_producers, labels=["profit"], type="meanvar")
    

        self.record('network', self.network.graph) 

        if self.t > self.p.simulation_steps:
            self.stop()


    def record_data(self, agents, labels=[], type="mean"):
        for label in labels:
            data=[agent.data[label][-1] for agent in agents]
            assert len(data)>0
            if type=="meanvar":
                self.record(label+"_mean",np.mean(data))
                self.record(label+"_var",np.var(data))
            elif type=="mean":
                self.record(label,np.mean(data))
            elif type=="sum":
                self.record(label,sum(data))
            else:
                raise ValueError("type not recognized")
            #print(f"Mean {label} : {np.mean(data)}")

 


    def update(self):
        
        # random.shuffle(self.energy_companies) #TODO CHECK DOES NOT FUCK UP WHICH ONE IS RE which oen is not
        for agent in self.agents:
            agent.update(self.t, self.agents)

  

    def end(self):
        """ Record evaluation measures at the end of the simulation. """
        #self.report('my_measure', 1)


class SchellingABM(ap.Agent):

    
    def setup(self):
        self.agent_type="consumer"
        self.config = self.p.consumer
        self.type = "citizen" if random.random() > self.config["ratio_industrial"] else "industrial"
        

    def update(self, t, agents):
   
        pass


