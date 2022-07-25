import parsl
from parsl import bash_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor
import os
import argparse

@bash_app
def run_main(args, out):
    return 'python main.py ' + args + ' &> ' + out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    if not os.path.isdir('outs'):
        os.mkdir('outs')
    
    parser.add_argument('--configs', type=str, nargs='+', help="Config files to run")
    parser.add_argument('--extra_args', type=str, help="Extra arguments to add to run")

    args = parser.parse_args()

    local_threads = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=len(args.configs), 
                label='local_threads'
            )
        ]
    )

    parsl.load(local_threads)

    jobs = []
    for config in args.configs:
        full_argset = '--config ' + config + ' ' + args.extra_args
        jobs.append(run_main(full_argset, out=os.path.join('outs', os.path.basename(config) + '.txt')))

    outputs = [job.result() for job in jobs]
