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


def run_experiment(config, module=None):

    if config["with_llm"]:
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

    if not config["varying_focus_parameter"]:
        print("***Run ABM model ***")
        model = ModelClass(config, id=expe_id)
        score = model.run()

    else:
        score = {}
        param = config["focus_parameter_abm"]  # what we are varying in the experiment
        for i, p in enumerate(config["parameters_abm"][param]):
            print("***Run ABM model with param {} ***".format(p))
            config["parameters_abm"][param] = p
            model = ModelClass(config, id=expe_id)
            score[i] = model.run()
        # TODO: save historics

        # -- score
        plot_final_score(score, y_label="Mean score", x_label=param, output_file=folder + "all_scores.png")

        # -- Visualise Final Results if any saved in the model
        if not config[
            "dev"
        ]:  # TODO: ADAPT TO OTHER MODELS, contains means that we only concatenate the images containing "grid" in their name
            concatenate_pngs(
                folder=folder, output_path=folder + "compare_all.png", title=f"ABM Expe, comparison on {param}", contains="grid"
            )
            concatenate_gifs(
                folder=folder, output_path=folder + "compare_all.gif", title=f"ABM Expe, comparison on {param}", contains="grid"
            )

    return score


def run_llm_experiment(config, module):
    """
    Run Experiment for LLM
    May make vary the parameter specified in config["focus_parameter_llm"] if the config say so
    """
    del config["parameters_abm"]

    expe_id = config["name"] + "/llm/" + str(datetime.datetime.now().strftime("%m%d%H%M%S"))
    folder = "outputs/" + expe_id + "/"
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    ModelClass = getattr(module, config["model_class_llm"], None)  # getattr(other_module, class_name, None)

    if not config["varying_focus_parameter"]:
        print("***Run LLM model ***")
        model = ModelClass(config, id=expe_id)
        score = model.run()
        model.save_historics(folder + "historics.json")

    else:
        score = {}
        param = config["focus_parameter_llm"]  # what we are varying in the experiment

        for i, p in enumerate(config["parameters_llm"][param]):
            print("***Run LLM model with {} param {} ***".format(param, p))
            config["parameters_llm"][param] = p
            model = ModelClass(config, id=expe_id)
            score[i] = model.run()
            model.save_historics(folder + "historics_{}.json".format(p))

        # -- score
        plot_final_score(score, y_label="Mean score", x_label=param, output_file=folder + "all_scores.png")

        # -- Visualise Final Results if any
        if not config[
            "dev"
        ]:  # TODO: ADAPT TO OTHER MODELS, contains means that we only concatenate the images containing "grid" in their name
            concatenate_pngs(
                folder=folder, output_path=folder + "compare_all.png", title=f"LLM Expe, comparison on {param}", contains="grid"
            )
            concatenate_gifs(
                folder=folder, output_path=folder + "compare_all.gif", title=f"LLM Expe, comparison on {param}", contains="grid"
            )

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
    args = parser.parse_args()
    xp = args.experiment

    # 1-- Load config file
    config_path = os.path.join(os.getcwd(), "config/" + xp + ".yml")
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(f"\nConfig:\n{config}")

    # 2-- Add open ai key to config
    with open(os.path.join(os.getcwd(), "config/openai.yml")) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
        config["openai_api_key"] = conf["openai_api_key"]

    # 3-- Run experiment
    # Here, several runs of the same model with different parameters are run.
    # Either it will launch a plain ABM experiment or a LLM-driven experiment, depending on the config file.
    if xp == "schelling":
        from src.models.schelling import model as schelling_model

        run_experiment(config, module=schelling_model)
    elif xp == "belief":
        from src.models.belief import model as belief_model

        run_experiment(config, module=belief_model)
    else:
        raise ValueError(f"Experiment {xp} not implemented yet")
