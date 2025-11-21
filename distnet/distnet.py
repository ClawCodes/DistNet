from typing import Callable, Union


from torch import nn, Tensor
from distnet.ring_allreduce import ring_reduce


class DistNet(nn.Module):
    def register_grad_hook(self, hook_func: Callable[[str, Tensor], Union[Tensor, None]]):
        named_params = dict(self.named_parameters())
        # Register gradient hooks
        '''
        for name, param in named_params.items():
            if param.requires_grad:
                def make_hook(param_name):  # bind hook to param name
                    def hook(grad):
                        new_grad = hook_func(param_name, grad)
                        return new_grad

                    return hook

                param.register_hook(make_hook(name))
        '''
        self.grad_params = [param for param in self.parameters() if param.requires_grad]
        for param in self.grad_params:
            def make_hook(parameter):  # bind hook to param name
                def hook(grad):
                    new_grad = hook_func(parameter, grad)
                    return new_grad

                return hook

            param.register_hook(make_hook(param))

