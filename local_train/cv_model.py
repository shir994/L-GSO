import subprocess
import shlex
import time
import os
import click


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--init_psi', type=str, default="0., 0.")
@click.option('--n_samples', type=int, default=10)
@click.option('--lr', type=float, default=1e-1)
@click.option('--step_data_gen', type=float, default=0.1)
@click.option('--cv', type=int, default=5)
def main(model, model_config_file, n_samples, lr, step_data_gen, cv, optimized_function, init_psi):
    command = "python end_to_end.py --model {0} --project_name cv_{2} \
    --work_space JhonDoe --model_config_file {1} --tags {0},{2},cv \
    --optimizer TorchOptimizer --optimized_function {2}  --init_psi {3} \
    --n_samples {4} --lr {5} --reuse_optimizer True --step_data_gen {6}"
    processes = []
    for _ in range(cv):
        command_pre = command.format(
                model,  # 0
                model_config_file,  # 1
                optimized_function,  # 2
                init_psi,  # 3
                n_samples,  # 4
                lr,  # 5
                step_data_gen,  # 6
            )
        print(command_pre)
        command_pre = shlex.split(command_pre)
        continue
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
