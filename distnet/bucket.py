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
        # guard to prevent double counting if hook is called multiple times for same param
        self.seen = set()
    def reset(self):
        
        self.ready_count = 0
        self.tensor.zero_()
        self.seen.clear()
        #print("RESET bucket done")

    def add_grad_tensor(self, p, g):
        pid = id(p)
        if pid in self.seen:
            return
        offset = self.offsets[pid]
        self.tensor[offset: offset + g.numel()].copy_(g.view(-1))
        self.ready_count += 1
        self.seen.add(pid)

    def is_ready(self):
        return self.ready_count == len(self.params)

    def scatter_to_params(self):
        #Scatter the reduced bucket back into parameters' grad value
        for p in self.params:
            offset=self.offsets[id(p)]
            if p.grad is None:
                # create grad tensor if needed (this is what PyTorch normally does for optimizers)
                p.grad = torch.empty_like(p)
            # copy reduced values into p.grad (view_as ensures shape matches)
            p.grad.copy_(self.tensor[offset : offset + p.numel()].view_as(p.grad))
       # print("SCATTER done")