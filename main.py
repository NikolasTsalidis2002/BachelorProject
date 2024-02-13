import datetime
import yaml
import os
from src.visualize import concatenate_pngs, concatenate_gifs, plot_final_score
import argparse
import pathlib

##########################################
#### CORE TODO ####
##########################################

# TODO: Many run for experiment, averaging over the runs
# TODO: Instantiate different relationships between agents

##########################################
#### RUNNING EXPERIMENT ####
##########################################


def run_experiment(config, module=None, with_llm=True):

    if with_llm:
        score = run_llm_experiment(config, module)
    else:
        score = run_abm_experiment(config, module)

    return score


def run_abm_experiment(config, module):
    """
    Run Experiment with ABM
    Make vary the parameter specified in config["main_abm"]
    """
    del config["parameters_llm"]

    # Initialise experiment
    expe_id = config["name"] + "/abm/" + str(datetime.datetime.now().strftime("%m%d%H%M%S"))
    folder = "outputs/" + expe_id + "/"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    ModelClass = getattr(module, config["model_class_abm"], None)  # getattr(other_module, class_name, None)

    print("***Run ABM model ***")
    model = ModelClass(config, id=expe_id)
    score = model.run()

    return score


def run_llm_experiment(config, module):
    """
    Run Experiment for LLM
    """
    del config["parameters_abm"]

    expe_id = config["name"] + "/llm/" + str(datetime.datetime.now().strftime("%m%d%H%M%S"))
    folder = "outputs/" + expe_id + "/"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    ModelClass = getattr(module, config["model_class_llm"], None)  # getattr(other_module, class_name, None)

    print("***Run LLM model ***")
    model = ModelClass(config, id=expe_id)
    score = model.run()
    model.save_historics(folder + "historics.json")

    return score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "-xp",
        "--experiment",
        type=str,
        choices=["schelling", "belief"],
        required=True,
        help="Name of the experiment to run (schelling or propagation currently implemented).",
    )
    parser.add_argument(
        "-a",
        "--agent_model",
        type=str,
        choices=["llm", "abm"],
        default="llm",
        help="Agent model: either LLM agent or simple ABM agent.",
    )
    args = parser.parse_args()
    xp = args.experiment
    with_llm= bool(args.agent_model=="llm")

    # 1-- Load config file
    config_path = os.path.join(os.getcwd(), "config/" + xp + ".yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(f"\nConfig:\n{config}")

    # 3-- Run experiment
    # Here, several runs of the same model with different parameters are run.
    # Either it will launch a plain ABM experiment or a LLM-driven experiment, depending on the config file.
    if xp == "schelling":
        from src.models.schelling import model as schelling_model
        run_experiment(config, module=schelling_model,with_llm=with_llm)
    elif xp == "belief":
        from src.models.belief import model as belief_model
        run_experiment(config, module=belief_model, with_llm=with_llm)
    else:
        raise ValueError(f"Experiment {xp} not implemented yet")
