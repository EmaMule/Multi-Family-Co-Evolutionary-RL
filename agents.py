import torch
from torch import nn
import numpy as np
from networks import *

NETWORK_TYPE = ClassicNet #change!
class NeuroAgentClassic:

    def __init__(self, input_shape, n_actions, use_softmax, mode = 'training'):
        assert mode in ['training', 'evaluating', 'deployng']
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.model = NETWORK_TYPE(input_shape, n_actions)
        self.use_softmax = use_softmax
        self.mode = mode


    # save model into file
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)


    # load model from file
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))


    def get_perturbable_layers(self):
      return [m for m in self.model.modules() if isinstance(m, nn.Linear) or isinstance(m, type(nn.Conv2d))]


    def get_perturbable_weights(self):
        weights = []
        for layer in self.get_perturbable_layers():
            weights.append(layer.weight.data.cpu().numpy().flatten())
            weights.append(layer.bias.data.cpu().numpy().flatten())
        return np.concatenate(weights)


    def set_perturbable_weights(self, flat_weights):
        idx = 0
        for layer in self.get_perturbable_layers():
            weight_size = layer.weight.numel()
            layer.weight.data = torch.tensor(flat_weights[idx: idx + weight_size].reshape(layer.weight.shape))
            idx += weight_size
            bias_size = layer.bias.numel()
            layer.bias.data = torch.tensor(flat_weights[idx: idx + bias_size].reshape(layer.bias.shape))
            idx += bias_size


    # mutate the model's weights by adding a normally distribute noise
    def mutate(self, std_dev):

        # get weights to mutate
        perturbable_weights = self.get_perturbable_weights()

        # generate the noise
        noise = np.random.normal(loc=0.0, scale=std_dev, size=perturbable_weights.shape).astype(np.float32)

        # apply the noise
        self.set_perturbable_weights(perturbable_weights + noise)

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
            elif self.mode == 'training' and not self.use_softmax:
                chosen_action = torch.argmax(masked_probs).item()
            elif self.mode == 'evaluating':
                chosen_action = torch.argmax(masked_probs).item()
            elif self.mode == 'deploying':
                chosen_action = torch.argmax(masked_probs).item()

            return chosen_action, masked_logits, masked_probs


    # get number of parameters
    def size(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters:", num_params)
        param_size_mb = num_params * 4 / (1024 ** 2)
        print(f"Model size: {param_size_mb:.2f} MB")
        return num_params
    
class DummyAgent:

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.mode = 'evaluating'

    def choose_action(self, inputs, action_mask):
        valid_actions = np.where(action_mask == 1)[0]
        masked_logits = torch.zeros(self.n_actions)
        masked_logits[action_mask == 0] = float('-inf')
        masked_probs = torch.ones(self.n_actions) / len(valid_actions)
        masked_probs[action_mask == 0] = 0
        return np.random.choice(valid_actions), masked_logits, masked_probs