# Neural Moment Matching Regression
Code for an upcoming NeurIPS submission

## How to Run codes?

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
