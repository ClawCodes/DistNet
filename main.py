from argparse import ArgumentParser
import subprocess
from typing import List
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 16
RUNS_DIR = PROJECT_ROOT / 'runs'
DEFAULT_OUTPUT_DIR = RUNS_DIR / 'latest'


def get_host_names() -> List[str]:
    result = subprocess.run(
        ["/bin/bash", "-c", "/usr/local/etc/emulab/tmcc hostnames | wc -l"],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()

    return output.split()


def get_num_workers() -> int:
    return len(get_host_names())

def main(args):
    # Validate worker size is feasible
    # init_process_pool
    # init base model
    # Broadcast model
    # init distributed sampler
    # broadcast run?
    ...

if __name__ == '__main__':
    # TODO: Determine if other params need to be created
        # Experiment type? (Need to review experiments)
    parser = ArgumentParser()
    parser.add_argument("-w", "--workers", help="Number of workers", type=int, default=get_num_workers())
    parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-e", "--epoch", help="Number of epochs to run", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("-o", "--output", help="Output directory", type=str, default=str(DEFAULT_OUTPUT_DIR))

    args = parser.parse_args()

    main(args)