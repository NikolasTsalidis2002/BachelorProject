import numpy as np
import random as rnd
import torch
from rewards import *
from training import train
import os
import yaml
import datetime
from model import EnergyModel
from utils import is_renewable, get_adjacency_matrix, sample_from_beta_distribution, extract_meso, extract_data, load_config
from visualize import animate, plot_energy_data
import pathlib


def simulate(config, visualize=True):
    model = EnergyModel(config)
    results = model.run(config["simulation_steps"])

    if visualize:
        run_visualisations(results, config)
    return results


def evaluate(config, solution, visualize=False, label=""):
    model = EnergyModel(config, agents_policy_weights=solution)
    results = model.run(config["simulation_steps"])

    if visualize:
        run_visualisations(results, config, label=label)

    return results


def learn(config, visualize=True):
    from rewards import reward_population

    ######## TRAIN AGENTS #########
    print("Training...")
    solution_best, solution_centroid, early_stopping_executed, logger = train(config["training"], config)

    # # Run a simulation with the best solution
    print("Running simulation with best solution...")
    for i in range(config["nb_evaluations"]):
        label = "best_solution_" + str(i)
        results = evaluate(config, solution_best, visualize=visualize, label=label)
        # fevals = reward(solution_best)
        # print(f"\nFirst run with best solution run 2:")
        # for i, agent_type in enumerate(config["learning_agents"]):
        #    print(f"{agent_type} reward: {fevals[i]}")

    # # Run a simulation with centroid solution
    print("Running simulation with centroid solution...")
    results = evaluate(config, solution_centroid, visualize=visualize, label="centroid")

    return results


def run_visualisations(results, config, label=""):

    # create dir
    if config["learning"]:
        folder = os.path.join(os.getcwd(), "experiments") + "/learn/" + config["date"] + "_" + label
    else:
        folder = os.path.join(os.getcwd(), "experiments") + "/abm/" + config["date"] + "_" + label

    print("About to save visu experiments at", folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    energy_data = extract_data(results.variables.EnergyModel, config, labels=["produced", "sold", "wasted"], meanvar=True)
    infrastructure_data = extract_data(results.variables.EnergyModel, config, labels=["num_plants", "num_active_plants"], meanvar=False)
    sustainability_data = extract_data(
        results.variables.EnergyModel,
        config,
        labels=["demand", "produced", "sold"],
        keys=[["RE", "NR"], ["RE", "NR"], ["RE", "NR"]],
        meanvar=False,
    )
    price_data = extract_data(
        results.variables.EnergyModel,
        config,
        labels=["profit", "energy_cost", "energy_prices"],
        keys=[[], config["energy"], ["RE", "NR"]],
        meanvar=False,
    )

    # TODO: ADD PEOPLE SUSTAINABILITY...
    plot_energy_data(data=energy_data, config=config, output_folder=folder, title="energy")
    plot_energy_data(data=infrastructure_data, config=None, output_folder=folder, title="infrastructure")
    plot_energy_data(data=sustainability_data, config=None, output_folder=folder, title="sustainability", labels=[["RE", "NR"]])
    plot_energy_data(
        data=price_data, config=None, output_folder=folder, title="Energy Economy", labels=[[], config["energy"], ["RE", "NR"]]
    )

    # ANIMATE
    network = results.variables.EnergyModel["network"]
    animate(
        data=results.variables.EnergyModel,
        network=network,
        energy_data=energy_data,
        sustainability_data=sustainability_data,
        infrastructure_data=infrastructure_data,
        config=config,
        output_folder=folder,
    )


# TODO Seeding for reproducibility
# np.random.seed(config["seed"])  # FIX so it can take none as a seed
# rnd.seed(config["seed"])

if __name__ == "__main__":

    config, tp = load_config()

    if config["learning"]:
        results = learn(config, visualize=True)
    else:
        results = simulate(config, visualize=True)
