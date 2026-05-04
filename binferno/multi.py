
import torch

class MultiModule():
    """
    A module that combines multiple feature extraction modules into a single module. 
    
    The output of each module is concatenated together to form the final feature vector.
    """
    def __init__(
        self,
        *args
    ):
        self.submodules = args

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        results = []
        for module in self.submodules:
            results.append(module(*args, **kwargs))

        return torch.cat(results, dim=1)