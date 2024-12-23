from trainer import Trainer
from agents import NeuroAgentClassic
from collections import deque
from copy import deepcopy
from utils import cosine_similarity
import numpy as np
class GeneticTrainer(Trainer):

    def __init__(self, env_type, render_mode, population_size, n_generations,
                 use_softmax, hof_size, dummy_size, dummy_decay_freq, n_elites, initial_std_dev,
                 min_std_dev, std_dev_decay, dissimilarity_weight, plot_eval_times,
                 plot_eval_freq, plot_eval_window, use_action_mask):

        super().__init__(env_type, render_mode, population_size, n_generations,
                        use_softmax, hof_size, dummy_size, dummy_decay_freq, initial_std_dev,
                         min_std_dev, std_dev_decay, dissimilarity_weight,
                         plot_eval_times, plot_eval_freq, plot_eval_window,
                        use_action_mask)

        self.n_elites = n_elites

        # randomly initialized elites (they are initialized with training but each time they are used we set the mode)
        self.elites = [NeuroAgentClassic(self.input_shape, self.n_actions, self.use_softmax) for _ in range(self.n_elites)]

        # hall of fame initially filled with the elites
        self.hof = deque([deepcopy(self.elites[i % self.n_elites]) for i in range(self.hof_size)], maxlen=self.hof_size)

        # winner is the last from the hall of fame
        self.winner = self.hof[-1]


    def train_step(self):

        # the first members of the population are the elites
        population = self.elites[:self.n_elites]

        # the others are mutations of the elites
        for i in range(self.population_size-self.n_elites):
            father_id = i % self.n_elites
            agent = deepcopy(self.elites[father_id])
            agent.mutate(self.std_dev)
            population.append(agent)

        # compute score for each member of the population
        rewards = np.zeros(self.population_size)
        for i in range(self.population_size):

            # against hall of fame
            for j in range(self.hof_size - self.dummy_size):
                hof_index = -1-j
                result = self.evaluate_agent(population[i], self.hof[hof_index])
                # enforce dissimilarity between the weights
                result -= self.dissimilarity_weight * cosine_similarity(self.hof[hof_index].model, population[i].model)
                rewards[i] += result

            # against dummies
            for j in range(self.dummy_size):
                result = self.evaluate_agent(population[i], self.dummy)
                rewards[i] += result

        rewards /= self.hof_size

        # update winner and hall of fame
        best_id = np.argmax(rewards)
        self.winner = deepcopy(population[best_id])
        self.record_play(self.winner, self.winner)
        self.hof.append(self.winner)

        # update the elites
        new_elite_ids = np.argsort(rewards)[-self.n_elites:] # maybe change
        for i in range(self.n_elites):
            id = new_elite_ids[i]
            self.elites[i] = population[id]