
import datetime
import pathlib


def test_abm_experiment(config, module):
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


def test_llm_experiment(config, module):
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