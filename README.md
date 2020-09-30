## Installing dependencies:
- We use the [comet.ml](https://www.comet.ml/site/) to log experiments. Please, register on the platform to be able to run our code.
- create an environment with conda yml file: ```conda env create -f conda_env.yml```.

In case you also want to check FFJORD you need to install

```pip install git+https://github.com/rtqichen/torchdiffeq```

## To reproduce experiments:
- activate environment with `conda activate lgso`.
- ```cd /L-GSO/local_train```. 
- Set the parameters in the correspondong `*_config.py` file. Execute the commands below for the particular experiment.

Dont forget to set the parameters of the used optimisers to the values, presented in the tables below.
(GAN, FFJORD, Void and LTS models).

You can find the exact commands to run experiments in the file: FILE. 


#### GAN surrogate parameters
The parameters are set in `gan_config.py`.

| Simulator Model | task | psi_dim| y_dim | x_dim | noise_dim | lr | batch_size | epochs | iters_discriminator | iters_generator | instance_noise_std | dis_output_dim | grad_penalty | gp_reg_coeff |
|:---:         |:---:         |    :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |          :---: |:---:         |     :---:      |
| Three Hump Model |CRAMER|2|1|2|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Rosenbrock 10dim  |CRAMER|10|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Rosenbrock submanifold 100dim |CRAMER|100|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Three Hump submanifold 40dim  |CRAMER|40|1|1|150|8e-4|512|15|1|1|0.01|256|True|10|x|
| Neural network optimisation |CRAMER|91|1|13|150|8e-4|512|15|1|1|0.01|256|True|10|x|


#### Void (LAX) baseline parameters:
The parameters are set in `void/void_config.py`.
`psi_dim, y_dim, x_dim` are set as above. The rest left unchanged.

#### Learning to simulate (LTS) baseline parameters:
The parameters are set in `learn_to_sim/lts_config.py`.
`y_dim, x_dim` are set as above. The rest left unchanged.

#### Optimizer parameters:
| Simulator Model | lr | torch_model | num_repetitions
|:---:         |     :---:      |          :---: |:---:         |
| Three Hump Model |0.1|Adam|10000|
| Rosenbrock 10dim  |0.1|Adam|10000|
| Rosenbrock submanifold 100dim |0.1|Adam|10000|
| Three Hump submanifold 40dim  |0.1|Adam|10000|
| Neural network optimisation |0.1|Adam|10000|

## To add your own function for optimisation:

Add a new class to `model.py`, inhereted from `YModel` and at the runtime, specify `--optimized_function YourClassName`. For example, see `Rosenbrock` class.

