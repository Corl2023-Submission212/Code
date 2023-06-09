# Hybrid Arm Dynamic Control

This repository is broken into four subfolders

1. Forward Model Development (forward_model_dev)
2. Generating Optimal Trajectories (generate_optimal_trajectories)
3. Control Policy Development (compare_policies)
4. Evaluation of Policies (dynamic_control_tests)

## 1. Forward Model Development

To train the forward models, run command

`python train_forward_models.py -model_name "model_name" -seeds number_of_seeds`

For example,
`python train_forward_models.py -model_name "LMU" -seeds`
 
The hyperparameters can be modified in the python file
 
To evaluate the forward models, run command
`python eval_forward_model.py -model_name "model_name" -seed seed_number`

For example,
`python eval_forward_model.py -model_name "LMU" -seed 0`

To view training plots, run command
`python aggregate_and_plot.py -logdirs "space separated list of models to plot" -output_file_name output_file_name -seeds number_of_seeds`

For example,
`python aggregate_and_plot.py -logdirs "LMU LSTM RNN MLP" -output_file_name full.svg -seeds 5`

## 2. Generating Optimal Trajectories

 To generate optimal trajectories, first start-end pairs must be constructed from the forward model dataset. To do so, run 
 python start_end_pairs.py
 
 Start_end_pairs used in the paper are already constructed
 
 To generate optimal trajectories, run
`python generate_trajectories_adam_lmu_full.py`
 
 This will generate the optimal trajectories for every start-end pair. 
 
 Optimal trajectories used in the paper are already provided.
 
## 3. Control Policy Development
 
 Using the optimal trajectories collected on the robot, the data must first be cleaned prior to policy training. This is done using the clean_policy_dataset_robot.py file. After the data is cleaned, the policies can be trained using the following command.
 
`python train.py -model_names "model_names" -seeds number_of_seeds`

For example
`python train.py -model_names "LMU LSTM" -seeds 1`

To view training plots, 
run command
`python aggregate_and_plot.py -logdirs "space separated list of models to plot" -output_file_name output_file_name -seeds number_of_seeds`

For example
`python aggregate_and_plot.py -logdirs "LMU LSTM RNN MLP" -output_file_name full.svg -seeds 5`

## 4. Evaluation of Policies

Dynamic control tests are divided into sections. The arbitrary poses in the workspace were used to test the policy's performance without and with loads. Additionally, a line trajectory was used to test the dynamic controllers ability to follow a line as various frequencies.

The data used to make the tables and plots are in the arbitrary and line folders. To recreate the figures and tables in the paper, the code in respective folders within eval can be used.




















