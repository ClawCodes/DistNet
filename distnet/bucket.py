import torch
class Bucket:
    def __init__(self, params):
        self.params = params
        # Offsets to map per-param grad to bucket
        self.offsets = []
        offset = 0
        bucket_size=0
        for p in params:
            n = p.numel()
            bucket_size += n
            self.offsets.append((offset, offset + n))
            offset += n
        self.size = bucket_size
        self.tensor = torch.zeros(self.size)
        self.ready_count = 0

    def add_grad(self, p):
        #Copy gradient into bucket
        idx = self.params.index(p)
        start, end = self.offsets[idx]
        self.tensor[start:end].copy_(p.grad.view(-1))
        self.ready_count += 1

    def is_ready(self):
        return self.ready_count == len(self.params)

    def scatter_to_params(self):
        #Scatter the reduced bucket back into parameters' grad value
        for p, (start, end) in zip(self.params, self.offsets):
            p.grad.copy_(self.tensor[start:end].view_as(p.grad))