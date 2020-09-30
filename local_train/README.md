## Example how to run

```
python end_to_end.py --init_psi 0.,0. --project_name my_first_project --work_space JhonDoe --tags gan, --model GANModel --model_config_file gan_config --optimizer NewtonOptimizer
```

All configuration of models could be done through `gan_config.py` and `optimizer_config.py` files.

`project_name` and `work_space` would be asked by program if not provided.

For now two type of models are available: `GANModel` and `FFJORDModel`.

Available optimizers: `GradientDescentOptimizer`, `NewtonOptimizer`, `LBFGSOptimizer`, `ConjugateGradientsOptimizer`(Polak–Ribiere version).