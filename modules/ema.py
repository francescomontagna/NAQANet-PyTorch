"""
Exponential Moving Average for model parameters.
"""


class EMA():

    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    # TODO study
    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]