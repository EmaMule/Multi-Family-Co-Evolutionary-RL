import torch
from torch import nn
import numpy as np

from networks import *

# agent using a neural network and working on classic environments of pettingzoo
class NeuroAgentClassic(nn.Module):

    def __init__(self, input_shape, n_actions, use_softmax, mode = 'training', network_type = ClassicNet):
        super(NeuroAgentClassic, self).__init__()
        assert mode in ['training', 'evaluating', 'deploying']
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.model = network_type(input_shape, n_actions)
        self.use_softmax = use_softmax
        self.mode = mode

        # disable gradient for the model
        for param in self.model.parameters():
            param.requires_grad = False


    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'n_actions': self.n_actions,
            'use_softmax': self.use_softmax,
            'mode': self.mode,
        }
        torch.save(checkpoint, filename)


    @classmethod
    def load(cls, filename):
        checkpoint = torch.load(filename, weights_only=False)
        instance = cls(
            input_shape=checkpoint['input_shape'],
            n_actions=checkpoint['n_actions'],
            use_softmax=checkpoint['use_softmax'],
            mode=checkpoint['mode']
        )
        instance.load_state_dict(checkpoint['model_state_dict'])
        return instance


    def get_perturbable_layers(self):
      return [m for m in self.model.modules() if isinstance(m, nn.Linear) or isinstance(m, type(nn.Conv2d))]


    def get_perturbable_weights(self):
        weights = []
        for layer in self.get_perturbable_layers():
            weights.append(layer.weight.data.cpu().numpy().flatten())
            if hasattr(layer, 'bias') and layer.bias is not None:
              weights.append(layer.bias.data.cpu().numpy().flatten())
        return np.concatenate(weights)


    def set_perturbable_weights(self, flat_weights):
        idx = 0
        for layer in self.get_perturbable_layers():
            weight_size = layer.weight.numel()
            layer.weight.data = torch.tensor(flat_weights[idx: idx + weight_size].reshape(layer.weight.shape))
            idx += weight_size
            if hasattr(layer, 'bias') and layer.bias is not None:
              bias_size = layer.bias.numel()
              layer.bias.data = torch.tensor(flat_weights[idx: idx + bias_size].reshape(layer.bias.shape))
              idx += bias_size


    # mutate the model's weights by adding a normally distribute noise
    def mutate(self, std_dev):

        # get weights to mutate
        perturbable_weights = self.get_perturbable_weights()

        # generate the noise
        noise = np.random.normal(loc=0.0, scale=std_dev, size=perturbable_weights.shape).astype(np.float32)

        weights = perturbable_weights + noise

        # apply the noise
        self.set_perturbable_weights(weights)

        return noise


    # choose best action or sample according to model's logits
    def choose_action(self, inputs, action_mask):

        self.model.eval()
        with torch.no_grad():

            # get action values
            logits = self.model(inputs).squeeze(0)
            masked_logits = logits.clone()
            masked_logits[action_mask == 0] = float('-inf')

            # get probabilities
            masked_probs = torch.nn.functional.softmax(masked_logits, dim=0)

            # choose action
            if self.mode == 'training' and self.use_softmax:
                chosen_action = torch.multinomial(masked_probs, 1).item()
            else:
                # mode = evaluating, mode = deploying and mode = training with not softmax
                chosen_action = torch.argmax(masked_probs).item()

            return chosen_action, logits, masked_logits, masked_probs


    # get number of parameters
    def size(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters:", num_params)
        param_size_mb = num_params * 4 / (1024 ** 2)
        print(f"Model size: {param_size_mb:.2f} MB")
        return num_params
    

# simple agent choosing random actions working on classic environments of pettingzoo
# (used as a baseline for evaluations)
class DummyAgent(nn.Module):

    def __init__(self, n_actions):
        super(DummyAgent, self).__init__()
        self.n_actions = n_actions
        self.mode = 'evaluating'


    def choose_action(self, inputs, action_mask):
        valid_actions = np.where(action_mask == 1)[0]
        logits = torch.zeros(self.n_actions)
        masked_logits = torch.zeros(self.n_actions)
        masked_logits[action_mask == 0] = float('-inf')
        masked_probs = torch.ones(self.n_actions) / len(valid_actions)
        masked_probs[action_mask == 0] = 0
        return np.random.choice(valid_actions), logits, masked_logits, masked_probs