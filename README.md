# Neural Maximum Moment Restriction
Code for a NeurIPS 2022 submission and fork from [Xu et al., 2021](https://github.com/liyuan9988/DeepFeatureProxyVariable). 

## How to Run Experiments

1. Install all Python dependencies
   ```
   pip install -r python_requirements.txt
   ```
2. Create an empty `dumps` directory (if needed) for capturing results 
   ```
   mkdir dumps
   ```
3. Run experiments
   ```
   python main.py <path-to-configs> ate
   ```
   
## Config files

The configuration files to replicate all results in the paper can be found in the `configs/` directory. For example, 
to replicate the results of the Demand experiment presented in Figure 2, you would run `main.py` for each JSON
file in `configs/figure2_config`.

## Reproducing our Figures

Reproducing our paper's Figures involves 3 steps: 1) running the experiments (described above), 2) aggregating the results,
and 3) running the visualization scripts.

**(2) Aggregation:** in the `data_visualization/results_aggregation_scripts/` folder,
there is a script for all required result aggregations for the paper. Each script
contains a docstring that describes its required inputs. Each script saves
the aggregated data to the `results/` folder.

**(3) Visualization/plotting:** in the `data_visualization/` folder, there are 6 R scripts
that reproduce the Figures in our paper using the aggregated data from (2). These scripts should
run self-contained; they need only have their working directory set to `results/`.

## Details on `main.py`

`main.py` is designed to be used from a command-line interface, as described above. Beneath the hood, main.py calls `main()`, which creates a time-stamped directory in `dumps/` to hold the current experiment's results. It then loads the user-specified json config file that should specify the experimental and model parameters. This config file is also saved to the experiment's results folder. 

Next, the `ate()` function is called. `ate()` passes the configuration dictionary and result's directory path to `experiment()` from src.experiment

`experiment()`: separates the config dict into `data_config` (specifying the data to be used for the experiment), `model_config` (specifying the model to be used for the experiment) and `n_repeat` (specifying the number of repetitions of the experiment to perform). The model's name (from model_config) is used by `get_run_func()` to retrieve the corresponding experiment execution function. Ex. our method has name "nmmr", causing `get_run_func()` to retrieve the execution function `NMMR_experiment()` from `src.models.NMMR.NMMR_experiments.py`. `grid_search_dict()` then uses the `data_config` to create a grid of experimental data parameters (i.e. if you specify a list of sample_sizes in your config file, each will be run during the experiment). A similar grid is created from the `model_config` parameters (e.g. if you want to run multiple variations of the same model in one experiment). And finally, a loop is executed over the grid of data_configs and model_configs, passing data parameters, model parameters, results directory path and experiment ID to the appropriate experiment execution function. The experiments are executed and results are saved to `result.csv` within the corresponding results directory under the  `dumps/` parent directory.
