# Neural Moment Matching Regression
Code for an upcoming NeurIPS submission   

## How to Run Experiments

1. Install all dependencies
   ```
   pip install -r requirements.txt
   ```
2. Create empty directories (if needed) for logging
   ```
   mkdir logs
   mkdir dumps
   ```
3. Run experiments
   ```
   python main.py <path-to-configs> <problem_setting>
   ```
   `<problem_setting>` can be selected from `ate` and `ope`, which corresponds to ate experiments and policy evaluation experiments in the paper by Xu et al. (https://arxiv.org/abs/2106.03907). Make sure to input a config file that corresponds correctly to each problem_setting. The results of each experiment can be found in the `dumps` folder. You can run in parallel by specifing  `-t` option.

## Details on `main.py`

`main.py` is designed to be used from a command-line interface, as described above. Beneath the hood, main.py calls `main()`, which creates a time-stamped directory in `dumps/` to hold the current experiment's results. It then loads the user-specified json config file that should specify the experimental and model parameters. This config file is also saved to the experiment's results folder. 

Next, the function specified by <problem_setting> is called (`ate` or `ope`). 

`ate()`: passes the configuration dictionary and result's directory path to `experiment()` from src.experiment

`ope()`: passes the configuration dictionary and result's directory path to `ope_experiments()` from src.src.experiment_ope

`experiment()`: separates the config dict into `data_config` (specifying the data to be used for the experiment), `model_config` (specifying the model to be used for the experiment) and `n_repeat` (specifying the number of repetitions of the experiment to perform). The model's name (from model_config) is used by `get_run_func()` to retrieve the corresponding experiment execution function. Ex. our method has name "nmmr", causing `get_run_func()` to retrieve the execution function `NMMR_experiment()` from `src.models.NMMR.NMMR_experiments.py`. `grid_search_dict()` then uses the `data_config` to create a grid of experimental data parameters (i.e. if you specify a list of sample_sizes in your config file, each will be run during the experiment). A similar grid is created from the `model_config` parameters (e.g. if you want to run multiple variations of the same model in one experiment). And finally, a loop is executed over the grid of data_configs and model_configs, passing data parameters, model parameters, results directory path and experiment ID to the appropriate experiment execution function. The experiments are executed and results are saved to `result.csv` within the corresponding results directory under the  `dumps/` parent directory.

`experiment_ope()`: identical to the `experiment()` function, except it has its own `get_run_func()` to call the appropriate experiment execution functions for the `ope` experiments.
