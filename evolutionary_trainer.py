from trainer import Trainer
from agents import NeuroAgentClassic
from collections import deque
from copy import deepcopy
from utils import cosine_similarity
import numpy as np

# EvolutionTrainer inherits from Trainer
class EvolutionTrainer(Trainer):

    def __init__(self, env_type, population_size, n_generations,
                 use_softmax, hof_size, dummy_size, dummy_decay_freq, initial_std_dev, min_std_dev,
                 std_dev_decay, dissimilarity_weight, learning_rate, plot_eval_times,
                 plot_eval_freq, plot_eval_window, use_action_mask, plot_path, video_folder):

        super().__init__(env_type, population_size, n_generations,
                         use_softmax, hof_size, dummy_size, dummy_decay_freq, initial_std_dev,
                         min_std_dev, std_dev_decay, dissimilarity_weight,
                         plot_eval_times, plot_eval_freq, plot_eval_window,
                         use_action_mask, plot_path, video_folder)

        self.learning_rate = learning_rate

        # randomly initialized starting agent (initialized with training but each time they are used we set the mode)
        self.winner = NeuroAgentClassic(self.input_shape, self.n_actions, self.use_softmax)

        # hall of fame filled with copies of the starting agent
        self.hof = deque([self.winner for _ in range(self.hof_size)], maxlen=self.hof_size)


    def train_step(self):

        population = []
        noises = []

        # populate with mutations of the current agent
        for i in range(self.population_size):
            agent = deepcopy(self.winner)
            noise = agent.mutate(self.std_dev)
            population.append(agent)
            noises.append(noise)

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

        # compute gradient
        gradient = np.zeros_like(noises[0])
        for i in range(self.population_size):
            gradient += noises[i] * rewards[i]
        gradient *= self.learning_rate/(self.population_size*self.std_dev)

        # update weights
        self.winner = deepcopy(self.winner)
        new_weights = self.winner.get_perturbable_weights() + gradient
        self.winner.set_perturbable_weights(new_weights)

        self.record_play(self.winner, self.winner)
        self.hof.append(self.winner)