from collections import deque
from copy import deepcopy
import numpy as np

from multitrainer import *
from utils import *
from agents import *


# specific trainer for genetic algorithm approach
class GeneticMultiTrainer(MultiTrainer):

    def __init__(self, env_type, n_families, family_size, n_generations, gamma, neg_multiplier,
                 use_softmax, family_hof_size, family_n_elites,
                 initial_std_dev, min_std_dev, std_dev_decay, plot_eval_times,
                 plot_eval_freq, plot_eval_window, use_action_mask, plot_path, video_folder, parallelization_type, network_type):

        super().__init__(env_type, n_families, family_size, n_generations, gamma, neg_multiplier,
                         use_softmax, family_hof_size, initial_std_dev,
                         min_std_dev, std_dev_decay, plot_eval_times, plot_eval_freq,
                         plot_eval_window, use_action_mask, plot_path, video_folder, parallelization_type, network_type)

        assert family_n_elites < family_size


        self.family_n_elites = family_n_elites
        self.hof_size = self.family_hof_size * self.n_families


    def initialize_train(self):

        # randomly initialized elites (they are initialized with training but each time they are used we set the mode)
        self.elites = []
        for i in range(self.n_families):
            family_elites = [NeuroAgentClassic(self.input_shape, self.n_actions, self.use_softmax, network_type = self.network_type) for _ in range(self.family_n_elites)]
            self.elites.append(family_elites)

        # hall of fame initially filled with the elites
        self.hof = deque([], maxlen = self.hof_size)
        j = 0
        while(len(self.hof) < self.hof_size):
            for i in range(self.n_families):
                model = deepcopy(self.elites[i][j % self.family_n_elites])
                self.hof.append(model)
            j += 1

        # winner is the last from the hall of fame
        self.winner = self.hof[-1]
        self.family_winners = []
        for i in range(self.n_families):
            hof_index = -1-i
            self.family_winners.append(self.hof[hof_index])


    def train_step(self):

        # the first members of the population are the elites
        families_population = []
        for i in range(self.n_families):
            family_population = self.elites[i][:self.family_n_elites]
            families_population.append(family_population)

        # the others are mutations of the elites
        for i in range(self.n_families):
            for j in range(self.family_size - self.family_n_elites):
                father_id = j % self.family_n_elites
                agent = deepcopy(self.elites[i][father_id])
                agent.mutate(self.std_dev)
                families_population[i].append(agent)

        # compute score for each member of the population
        families_rewards = self.schedule_parallel_training(families_population=families_population)

        # update winner and hall of fame
        best_rewards = np.zeros(self.n_families)
        for i in range(self.n_families):
            best_id = np.argmax(families_rewards[i])
            best_rewards[i] = np.max(families_rewards[i])
            self.family_winners[i] = deepcopy(families_population[i][best_id])
            self.families_train_rewards[i].append(best_rewards[i])
            self.hof.append(self.family_winners[i])

        best_family = np.argmax(best_rewards)
        self.winner = self.family_winners[best_family]
        self.record_play(self.winner, self.winner)

        # update the elites
        for i in range(self.n_families):
            new_elite_ids = np.argsort(families_rewards[i])[-self.family_n_elites:]
            for j in range(self.family_n_elites):
                id = new_elite_ids[j]
                self.elites[i][j] = families_population[i][id]