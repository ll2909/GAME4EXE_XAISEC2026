import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleGradients:
    def __init__(self, model, multiply = False, differentiable = True):
        self.model = model,
        self.multiply = multiply
        self.differentiable = differentiable

    
    def attribute(self, input : torch.Tensor):
        
        if not input.requires_grad:
            input = input.clone().detach().requires_grad_(True)

        gradient = torch.autograd.grad(outputs=self.model[0](input), inputs=input, create_graph=self.differentiable)[0]

        if self.multiply:
            gradient = gradient * input
        
        return gradient