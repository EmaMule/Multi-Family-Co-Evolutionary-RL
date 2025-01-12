from collections import deque
from copy import deepcopy
import numpy as np

from multitrainer import *
from utils import *
from agents import *


# specific trainer for evolutionary strategy approach
class EvolutionMultiTrainer(MultiTrainer):

    def __init__(self, env_type, n_families, family_size, n_generations, gamma, neg_multiplier, normalize_gradient,
                 use_softmax, family_hof_size, initial_std_dev, min_std_dev,
                 std_dev_decay, learning_rate, plot_eval_times, plot_eval_freq,
                 plot_eval_window, use_action_mask, plot_path, video_folder, parallelization_type, network_type):

        super().__init__(env_type, n_families, family_size, n_generations, gamma, neg_multiplier,
                         use_softmax, family_hof_size, initial_std_dev,
                         min_std_dev, std_dev_decay, plot_eval_times, plot_eval_freq,
                         plot_eval_window, use_action_mask, plot_path, video_folder, parallelization_type, network_type)

        self.learning_rate = learning_rate
        self.hof_size = self.n_families * self.family_hof_size
        self.normalize_gradient = normalize_gradient


    def initialize_train(self):

        # randomly initialized starting agent (initialized with training but each time they are used we set the mode)
        self.family_winners = []
        for i in range(self.n_families):
            self.family_winners.append(NeuroAgentClassic(self.input_shape, self.n_actions, self.use_softmax, network_type = self.network_type))

        self.winner = self.family_winners[0]

        # hall of fame filled with copies of the starting agent
        self.hof = deque([], maxlen = self.hof_size)
        while(len(self.hof) < self.hof_size):
            for i in range(self.n_families):
                model = deepcopy(self.family_winners[i])
                self.hof.append(model)


    def train_step(self):

        families_population = []
        families_noises = []

        for i in range(self.n_families):

            family_population = []
            family_noises = []

            # populate with mutations of the current agent
            for j in range(self.family_size):
                agent = deepcopy(self.family_winners[i])
                noise = agent.mutate(self.std_dev)
                family_population.append(agent)
                family_noises.append(noise)

            families_population.append(family_population)
            families_noises.append(family_noises)


        # compute score for each member of the population
        families_rewards = self.schedule_parallel_training(families_population=families_population)

        mean_family_rewards = np.array([np.mean(i) for i in families_rewards])
        for i in range(self.n_families):
            self.families_train_rewards[i].append(mean_family_rewards[i])
            if self.normalize_gradient: # TRY TO NORMALIZE
              families_rewards[i] -= mean_family_rewards[i]

        # compute gradients
        gradients = []
        for i in range(self.n_families):
            gradient = np.zeros_like(families_noises[i][0])
            for j in range(self.family_size):
                # if not self.normalize_gradient or families_rewards[i][j] > 0:
                gradient += families_noises[i][j] * families_rewards[i][j]
            gradient *= self.learning_rate / (self.family_size * self.std_dev)
            gradients.append(gradient)

        # update weights
        for i in range(self.n_families):
            self.family_winners[i] = deepcopy(self.family_winners[i])
            new_weights = self.family_winners[i].get_perturbable_weights() + gradients[i]
            self.family_winners[i].set_perturbable_weights(new_weights)
            self.hof.append(self.family_winners[i])

        # ESTIMATE THE BEST FAMILY
        best_family = np.argmax(mean_family_rewards)
        self.winner = self.family_winners[best_family]
        self.record_play(self.winner, self.winner)