import subprocess
import numpy as np
import shlex
import time
import os
import click


@click.command()
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--init_psi', type=str, default="0., 0.")
def main(optimized_function, init_psi):
    ns = [3, 5, 7, 9]
    hs = np.logspace(-3, 0, 19)
    command = "python baseline.py --project_name grid_search_num_diff_{0} \
    --work_space USERNAME --tags n_{2},h_{3},grid_search  \
    --n {2} --h {3} --optimizer_config_file  optimizer_config_num_diff \
    --optimizer TorchOptimizer --optimized_function {0}  --init_psi {1}"
    processes = []
    for n in ns:
        for h in hs:
            command_pre = command.format(
                    optimized_function,  # 0
                    init_psi,  # 1
                    n,  # 2
                    h,  # 3
                )
            print(command_pre)
            command_pre = shlex.split(command_pre)
            print(command_pre)
            process = subprocess.Popen(command_pre,
                                       shell=False,
                                       close_fds=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,
                                       preexec_fn=os.setsid)
            processes.append(process)
            time.sleep(5.)

    for process in processes:
        print(process.pid)


if __name__ == "__main__":
    main()
