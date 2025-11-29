import os
from argparse import ArgumentParser
import subprocess
from typing import Optional
from pathlib import Path

import torch
import torch.distributed as dist

from distnet.localnet import DistLocalNet
from distnet.localnet import DistCNN
from distnet.util import load_cifar10, distributed_train, distributed_test, broadcast_model

PROJECT_ROOT = Path(__file__).parent

print(PROJECT_ROOT)

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 16
DEFAULT_BUCKET_SIZE = 5
DEFAULT_MODEL = 'cnn'  # 'fc' or 'cnn'
RUNS_DIR = PROJECT_ROOT / 'runs'
DEFAULT_OUTPUT_DIR = RUNS_DIR / 'latest'


def get_num_workers() -> int:
    result = subprocess.run(
        ["/bin/bash", "-c",
         r"""geni-get -a | \
            grep -Po '<interface_ref client_id=\\".*?\"' | \
            sed 's/<interface_ref client_ide=\\"\(.*\)\\"/\1/' | \
            sort | \
            uniq | \
            wc -l
         """
         ],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()

    return int(output)

def main(args) -> None:
    dist.init_process_group(backend='gloo')

    # Select model based on argument
    if args.model == 'fc':
        net = DistLocalNet(bucket_size=args.bucket_size)
    elif args.model == 'cnn':
        net = DistCNN(bucket_size=args.bucket_size)
    else:
        raise ValueError(f"Unknown model type: {args.model}. Use 'fc' or 'cnn'")

    net.register_grad_hook(net.dist_hook)

    # broadcast parameters from rank 0 to other nodes
    dist.barrier()
    broadcast_model(net, src=0)
    dist.barrier()

    train_loader, test_loader = load_cifar10(args.batch_size)

    output_dir = Path(args.output)

    fname = f"node{dist.get_rank()}_batch_{args.batch_size}_epochs_{args.epoch}_buckets_{args.bucket_size}.json"
    outfile = output_dir / fname
    if output_dir != DEFAULT_OUTPUT_DIR:
        outfile = RUNS_DIR / output_dir / fname

    os.makedirs(outfile.parent, exist_ok=True)

    # Pass test_loader to evaluate accuracy after each epoch 
    # for convergence analysis
    distributed_train(net, train_loader, test_loader, args.batch_size, epochs=args.epoch, outfile=outfile)

    distributed_test(net, test_loader, outfile)
   
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch-size", help="Batch size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-e", "--epoch", help="Number of epochs to run", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("-u", "--bucket-size", help="Max size of buckets in mb", type=int, default=DEFAULT_BUCKET_SIZE)
    parser.add_argument("-o", "--output", help="Output directory", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("-m", "--model", help="Model type: 'fc' or 'cnn'", type=str, default=DEFAULT_MODEL, choices=['fc', 'cnn'])

    args = parser.parse_args()

    main(args)