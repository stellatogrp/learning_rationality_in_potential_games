import sys
import hydra
import pickle
import datetime

from src.experiments.activeset_experiment import run_full_activeset_experiment, create_plots_activeset
from src.experiments.choice_experiment import run_full_choice_experiment, create_plots_choice
from src.experiments.cournot_experiment import run_full_cournot_experiment, create_plots_cournot
from src.experiments.cournot_gurobi import run_full_cournot_gurobi_experiment, create_plots_cournot_gurobi


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='congestion_main_experiment_run.yaml')
def run_toy(cfg):
    if cfg['prev_save'] is None:
        results = run_full_activeset_experiment(cfg)
        log(results)
        create_plots_activeset(results, f"{sys.argv[1]}/plot"[14:], cfg)
    else:
        file = open(sys.argv[1][14:-27] + cfg['prev_save'] + '/result.pickle', "rb")
        results = pickle.load(file)
        file.close()
        log(results)
        create_plots_activeset(results, f"{sys.argv[1]}/plot"[14:], cfg)


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='cournot_run.yaml')
def run_cournot(cfg):
    if cfg['prev_save'] is None:
        results = run_full_cournot_experiment(cfg)
        log(results)
        create_plots_cournot(results, f"{sys.argv[1]}/plot"[14:], cfg)
    else:
        file = open(sys.argv[1][14:-27] + cfg['prev_save'] + '/result.pickle', "rb")
        results = pickle.load(file)
        file.close()
        log(results)
        create_plots_cournot(results, f"{sys.argv[1]}/plot"[14:], cfg)


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='cournot_gurobi_run.yaml')
def run_cournot_gurobi(cfg):
    if cfg['prev_save'] is None:
        results = run_full_cournot_gurobi_experiment(cfg)
        log(results)
        create_plots_cournot_gurobi(results, f"{sys.argv[1]}/plot"[14:], cfg)
    else:
        file = open(sys.argv[1][14:-27] + cfg['prev_save'] + '/result.pickle', "rb")
        results = pickle.load(file)
        file.close()
        log(results)
        create_plots_cournot_gurobi(results, f"{sys.argv[1]}/plot"[14:], cfg)


@hydra.main(version_base=None, config_path='configs/run_cfgs/', config_name='choice_run.yaml')
def run_choice(cfg):
    if cfg['prev_save'] is None:
        results = run_full_choice_experiment(cfg)
        log(results)
        create_plots_choice(results, f"{sys.argv[1]}/plot"[14:], cfg)
    else:
        file = open(sys.argv[1][14:-27] + cfg['prev_save'] + '/result.pickle', "rb")
        results = pickle.load(file)
        file.close()
        log(results)
        create_plots_choice(results, f"{sys.argv[1]}/plot"[14:], cfg)


def log(results):
    with open(f"{sys.argv[1]}/result.pickle"[14:], "wb") as fp:
        pickle.dump(results, fp)


if __name__ == "__main__":
    base = 'hydra.run.dir=outputs'

    if sys.argv[1] == 'toy':
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        run_toy()

    if sys.argv[1] == "cournot":
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        run_cournot()

    if sys.argv[1] == "cournot_gurobi":
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        run_cournot_gurobi()

    if sys.argv[1] == "choice":
        sys.argv[1] = base + '/' + sys.argv[1] + f'/{datetime.date.today()}/{str(datetime.datetime.now())[:-10]}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        run_choice()