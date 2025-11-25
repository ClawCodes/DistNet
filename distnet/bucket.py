import torch
class Bucket:
    def __init__(self, params):
        self.params = params
        self.param_ids = set()
        # Offsets to map per-param grad to bucket
        self.offsets = {}
        offset = 0
        for p in params:
            self.param_ids.add(id(p))
            self.offsets[id(p)]=offset
            offset += p.numel()
        self.size = offset
        self.tensor = torch.zeros(self.size)
        self.ready_count = 0

    def reset(self):
        self.ready_count = 0
        self.tensor.zero_()
        
    def add_grad(self, p):
        #Copy gradient into bucket
        offset=self.offsets[id(p)]
        self.tensor[offset: offset + p.numel()].copy_(p.view(-1))
        self.ready_count += 1

    def is_ready(self):
        return self.ready_count == len(self.params)

    def scatter_to_params(self):
        #Scatter the reduced bucket back into parameters' grad value
        for p in self.params:
            offset=self.offsets[id(p)]
            p.copy_(self.tensor[offset : offset + p.numel()].view_as(p))