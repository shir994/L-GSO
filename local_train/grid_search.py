import subprocess
import numpy as np
import shlex
import time
import os
import click


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--init_psi', type=str, default="0., 0.")
def main(model, model_config_file, optimized_function, init_psi):
    psi_dim = len([float(x.strip()) for x in init_psi.split(',')])
    step_data_gens = [0.05, 0.1, 0.5, 1., 5.]
    if optimized_function == "RosenbrockModelDegenerateInstrict":
        n_samples_search = [20, 50, 100, 200]
    else:
        n_samples_search = [int(np.ceil(psi_dim / 4)), int(np.ceil(psi_dim / 2)), psi_dim, 2 * psi_dim]
    command = "python end_to_end.py --model {0} --project_name grid_search_{2} \
    --work_space JhonDoe --model_config_file {1} --tags {0},{2},grid_search \
    --optimizer TorchOptimizer --optimized_function {2}  --init_psi {3} \
    --n_samples {4} --lr 0.1 --step_data_gen {5} --reuse_optimizer True"
    processes = []
    for step_data_gen in step_data_gens:
        for n_samples in n_samples_search:
            command_pre = command.format(
                    model,  # 0
                    model_config_file,  # 1
                    optimized_function,  # 2
                    init_psi,  # 3
                    n_samples,  # 4
                    step_data_gen,  # 5
                )

            print(command_pre)
            continue
            command_pre = shlex.split(command_pre)
            print(command_pre)
            process = subprocess.Popen(command_pre,
                                       shell=False,
                                       close_fds=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       preexec_fn=os.setsid)
            processes.append(process)
            time.sleep(30.)

    for process in processes:
        print(process.pid)

if __name__ == "__main__":
    main()
