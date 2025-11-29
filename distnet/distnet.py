from typing import Callable, Union
from torch import nn, Tensor
from distnet.bucket import Bucket

from distnet.ring_allreduce import ring_reduce

class DistNet(nn.Module):
    def __init__(self, bucket_size: int):
        super().__init__()
        self.grad_params = [param for param in self.parameters() if param.requires_grad]

        # Setup buckets
        bucket_cap_mb = bucket_size
        cap_bytes = bucket_cap_mb * 1024 * 1024
        self.buckets = []
        curr_bucket = []
        curr_bytes = 0

        for param in self.grad_params:
            size = param.numel() * param.element_size()
            if curr_bucket and (curr_bytes + size > cap_bytes):
                # finalize current bucket
                self.buckets.append(Bucket(list(curr_bucket)))
                curr_bucket, curr_bytes = [], 0
            curr_bucket.append(param)
            curr_bytes += size

        # add last bucket if any
        if curr_bucket:
            self.buckets.append(Bucket(list(curr_bucket)))

        # Set for quickly looking up bucket for a parameter
        self.param_id_to_bucket = {}
        for b in self.buckets:
            for p in b.params:
                self.param_id_to_bucket[id(p)] = b

        # Counters for debugging
        # self.hookFireCount = 0
        # self.reduceFireCount = 0

    def dist_hook(self, parameter, grad):
        # self.hookFireCount += 1

        bucket = self.param_id_to_bucket.get(id(parameter))
        if bucket is None:
            # shouldn't happen if buckets constructed correctly
            raise RuntimeError("Parameter not found in any bucket")
        # Add gradient to the bucket
        bucket.add_grad_tensor(parameter, grad)
        if bucket.is_ready():
            ring_reduce(bucket.tensor)
            bucket.scatter_to_params()
        return grad

    def reset_buckets(self):
        for b in self.buckets:
            b.reset()

    def register_grad_hook(self, hook_func: Callable[[str, Tensor], Union[Tensor, None]]):
        for param in self.grad_params:
            def make_hook(parameter):  # bind hook to param name
                def hook(grad):
                    new_grad = hook_func(parameter, grad)
                    return new_grad
                return hook
            param.register_hook(make_hook(param))


