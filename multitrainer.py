from pettingzoo.classic import tictactoe_v3, connect_four_v3, texas_holdem_no_limit_v6
import imageio
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import clear_output
from abc import ABC, abstractmethod
import os
import math

from utils import *
from agents import *


# generic class implementing methods used by both approaches (enabled to work
# with many 'families' and to train using multiple parallel processes)
class MultiTrainer(ABC):

    def __init__(self, env_type, n_families, family_size, n_generations, gamma, neg_multiplier,
                 use_softmax, family_hof_size, initial_std_dev,
                 min_std_dev, std_dev_decay, plot_eval_times, plot_eval_freq,
                 plot_eval_window, use_action_mask, plot_path, video_folder, parallelization_type, network_type):

        assert network_type in ['ClassicNet', 'DeepNet']
        assert env_type in ['tictactoe_v3', 'connect_four_v3', 'texas_holdem_no_limit_v6']
        assert parallelization_type in ['family', 'hof', 'no']

        # create workers if using parallelization (global becuase can't use self)
        if parallelization_type != 'no':
            global WORKERS
            WORKERS = Pool(max_workers = os.cpu_count())

        # training parameters
        self.gamma = gamma
        self.neg_multiplier = neg_multiplier
        self.use_softmax = use_softmax
        self.n_families = n_families
        self.family_size = family_size
        self.family_hof_size = family_hof_size
        self.n_generations = n_generations
        self.initial_std_dev = initial_std_dev
        self.min_std_dev = min_std_dev
        self.std_dev_decay = std_dev_decay
        self.use_action_mask = use_action_mask
        self.env_type = env_type
        self.parallelization_type = parallelization_type

        #choice of the network
        if network_type == 'ClassicNet':
          self.network_type = ClassicNet
        elif network_type == 'DeepNet':
          self.network_type = DeepNet

        # parameters depending on the environment
        self.render_mode = 'rgb_array' # rendering mode
        if self.env_type == 'tictactoe_v3':
            self.input_shape = [3,3,2]
            self.n_actions = 9
            self.players = ['player_1', 'player_2']
        elif self.env_type == 'connect_four_v3':
            self.input_shape = [6,7,2]
            self.n_actions = 7
            self.players = ['player_0', 'player_1']
        elif self.env_type == 'texas_holdem_no_limit_v6':
            self.input_shape = [54]
            self.n_actions = 5
            self.players = ['player_0', 'player_1']

        # plot parameters
        self.dummy = DummyAgent(self.n_actions)
        self.plot_eval_window = plot_eval_window
        self.plot_eval_times = plot_eval_times
        self.plot_eval_freq = plot_eval_freq
        self.plot_path = plot_path

        # video parameters
        self.video_folder = video_folder

        # initialize variables
        self.families_eval_rewards = [[] for i in range(self.n_families)]
        self.families_train_rewards = [[] for i in range(self.n_families)]
        self.families_mean_eval_rewards = [[] for i in range(self.n_families)]
        self.step_count = 0
        self.std_dev = self.initial_std_dev
        self.winner = None
        self.family_winners = [None for i in range(self.n_families)]


    def transform_obs(self, obs):
        if self.env_type == 'tictactoe_v3':
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        elif self.env_type == 'connect_four_v3':
            obs = obs.permute(2, 0, 1).unsqueeze(0)
        elif self.env_type == 'texas_holdem_no_limit_v6':
            obs = obs.unsqueeze(0)
        return obs


    def initialize_env(self):
        if self.env_type == 'tictactoe_v3':
          env = tictactoe_v3.env(render_mode=self.render_mode)
        elif self.env_type == 'connect_four_v3':
          env = connect_four_v3.env(render_mode=self.render_mode)
        elif self.env_type == 'texas_holdem_no_limit_v6':
          env = texas_holdem_no_limit_v6.env(render_mode=self.render_mode)
        env.reset()
        return env


    # for parallel training at family level
    def evaluate_family(self, family):

        rewards = np.zeros(self.family_size)
        for j in range(self.family_size):

            for k in range(self.hof_size):
                hof_index = -1-k
                result = self.evaluate_agent(family[j], self.hof[hof_index], True)
                rewards[j] += result

        rewards /= self.hof_size
        return rewards


    # for parallel training at agent level
    def evaluate_against_hof(self, agent):

        reward = 0
        for k in range(self.hof_size):
            hof_index = -1-k
            result = self.evaluate_agent(agent, self.hof[hof_index], True)
            reward += result

        reward /= self.hof_size
        return reward


    def schedule_parallel_training(self, families_population):

      if self.parallelization_type == 'family':
        for i in range(self.n_families):
            WORKERS.submit_task(self.evaluate_family, families_population[i])
        families_rewards = WORKERS.collect_results()

      elif self.parallelization_type == 'hof':
        for i in range(self.n_families):
            for j in range(self.family_size):
                WORKERS.submit_task(self.evaluate_against_hof, families_population[i][j])
        families_rewards = WORKERS.collect_results()
        families_rewards = np.reshape(families_rewards, (self.n_families, self.family_size))

      elif self.parallelization_type == 'no':
        families_rewards = []
        for i in range(self.n_families):
            for j in range(self.family_size):
                reward = self.evaluate_against_hof(families_population[i][j])
                families_rewards.append(reward)
        families_rewards = np.reshape(families_rewards, (self.n_families, self.family_size))

      return families_rewards


    def play_game(self, agent1, agent2, save_video = False):

        env = self.initialize_env()

        if save_video:
            path = self.video_folder + "/epoch_"+str(self.step_count)
            frames = []
            self.start_log(path)

        total_rewards = [0, 0]
        agents = [agent1, agent2]
        steps = 0

        for player in env.agent_iter():  # AEC mode!

            observation, reward, termination, truncation, _ = env.last()
            done = termination or truncation
            steps += 1

            player_id = self.players.index(player)
            total_rewards[player_id] += reward

            if done:
                action = None

            else:

                mask = torch.tensor(observation["action_mask"], dtype=torch.uint8)
                obs = torch.tensor(observation['observation'], dtype=torch.float32)
                obs = self.transform_obs(obs)

                # an agent can do the wrong action in 'training'
                if not self.use_action_mask and agents[player_id].mode == 'training':
                    mask = torch.ones_like(mask)

                action, logits, mlogits, probs = agents[player_id].choose_action(obs, mask)

            env.step(action)

            # save the rendered frame for the video and write log
            if save_video:
                frame = env.render()
                frames.append(frame)
                self.write_log(path, agents[player_id], player, action, logits, mlogits, probs)

        env.close()

        if save_video:
            self.compose_video(path, frames)

        return total_rewards, steps-2


    def custom_reward(self, reward, steps):
        reward = reward * (self.gamma ** steps)
        if reward < 0:
            reward = reward * self.neg_multiplier
        return reward


    # evaluate the agent by playing against the evaluator
    def evaluate_agent(self, agent, evaluator, use_custom_reward, times = 1):

        total_reward = 0
        agent.mode = 'training'
        evaluator.mode = 'evaluating'

        for i in range(times):

            rewards, steps = self.play_game(evaluator, agent)
            reward = self.custom_reward(rewards[1], steps) if use_custom_reward else rewards[1]
            total_reward += reward

            rewards, steps = self.play_game(agent, evaluator)
            reward = self.custom_reward(rewards[0], steps) if use_custom_reward else rewards[0]
            total_reward += reward

        return total_reward / times


    def record_play(self, agent1, agent2):
        agent1.mode = 'deploying'
        agent2.mode = 'deploying'
        self.play_game(agent1, agent2, save_video = True)


    # make the human play against the agent
    def play_against(self, agent, start_first = True):

        env = self.initialize_env()

        agent.mode = 'deploying'

        path = './play_against'
        self.start_log(path)
        frames = []
        total_rewards = [0, 0]
        steps = 0
        user_position = 0 if start_first else 1

        for player in env.agent_iter():  # AEC mode!

            observation, reward, termination, truncation, _ = env.last()
            done = termination or truncation
            steps += 1

            player_id = self.players.index(player)
            total_rewards[player_id] += reward

            # visualize current state
            frame = env.render()
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(frame)
            ax.axis('off')
            plt.show()

            if done:
                action = None

            else:

                mask = torch.tensor(observation["action_mask"], dtype=torch.uint8)
                obs = torch.tensor(observation['observation'], dtype=torch.float32)
                obs = self.transform_obs(obs)

                frame = env.render()
                frames.append(frame)

                # human's turn
                if user_position == player_id:

                    # choose action
                    print("Your Turn! Choose an action between 0 and", self.n_actions - 1)
                    print("Valid actions:", [i for i, valid in enumerate(mask) if valid])
                    while True:
                        try:
                            print("Enter your action: ")
                            action = int(input())
                            if action in [i for i, valid in enumerate(mask) if valid]:
                                break
                            else:
                                print("Invalid action. Please choose a valid action.")
                        except ValueError:
                            print("Invalid input. Please enter an integer.")

                # agent's turn
                else:

                    # choose action
                    action, logits, mlogits, probs = agent.choose_action(obs, mask)
                    self.write_log(path, agent, player, action, logits, mlogits, probs)

            env.step(action)

        # visualize current state
        frame = env.render()
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame)
        ax.axis('off')
        plt.show()

        env.close()

        print("Game over!")
        print("You won!" if total_rewards[user_position] > total_rewards[1-user_position] else "You lose!")

        self.compose_video(path, frames)

        return total_rewards, steps-2


    # make the agent evaluate each configuration of a game
    def evaluate_with_agent(self, agent):

        env = self.initialize_env()

        agent.mode = 'deploying'

        path = './evaluate_with_agent'
        self.start_log(path)
        frames = []

        for player in env.agent_iter():  # AEC mode!

            observation, _, termination, truncation, _ = env.last()
            done = termination or truncation

            # visualize current state
            frame = env.render()
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(frame)
            ax.axis('off')
            plt.show()

            if done:
                action = None

            else:

                mask = torch.tensor(observation["action_mask"], dtype=torch.uint8)
                obs = torch.tensor(observation['observation'], dtype=torch.float32)
                obs = self.transform_obs(obs)

                frame = env.render()
                frames.append(frame)

                # agent evaluation
                action, logits, mlogits, probs = agent.choose_action(obs, mask)
                self.write_log(path, agent, player, action, logits, mlogits, probs)

                # human chooses action
                print("Your Turn! Choose an action between 0 and", self.n_actions - 1)
                print("Valid actions:", [i for i, valid in enumerate(mask) if valid])
                while True:
                    try:
                        print("Enter your action: ")
                        action = int(input())
                        if action in [i for i, valid in enumerate(mask) if valid]:
                            break
                        else:
                            print("Invalid action. Please choose a valid action.")
                    except ValueError:
                        print("Invalid input. Please enter an integer.")

            env.step(action)

        # visualize current state
        frame = env.render()
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(frame)
        ax.axis('off')
        plt.show()

        env.close()

        print("Game over!")

        self.compose_video(path, frames)


    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def initialize_train(self):
        pass


    def train(self):

        # initialize variables
        self.families_eval_rewards = [[] for i in range(self.n_families)]
        self.families_train_rewards = [[] for i in range(self.n_families)]
        self.families_mean_eval_rewards = [[] for i in range(self.n_families)]
        self.step_count = 0
        self.std_dev = self.initial_std_dev
        self.initialize_train()

        # training loop
        for t in tqdm(range(self.n_generations)):

            # train
            self.train_step()

            # evaluation
            rewards  = []
            for i in range(self.n_families):
                reward = self.evaluate_agent(self.family_winners[i], self.dummy, False, self.plot_eval_times)
                rewards.append(reward)
            self.update_metrics(rewards)

            if self.step_count % self.plot_eval_freq == 0 and self.step_count != 0:
                self.plot_rewards()
                self.plot_collected_rewards()

            self.step_count+=1
            self.std_dev = max(self.min_std_dev, self.std_dev * self.std_dev_decay)


    def update_metrics(self, rewards):
        for i in range(self.n_families):
            self.families_eval_rewards[i].append(rewards[i])
            n_elems = min(len(self.families_eval_rewards[i]), self.plot_eval_window)
            reward_window = self.families_eval_rewards[i][-n_elems:]
            self.families_mean_eval_rewards[i].append(np.mean(reward_window))


    def plot_rewards(self):
        # Ensure the colors are defined based on the number of families
        colors = cm.tab10.colors if self.n_families <= 10 else cm.get_cmap('tab20', self.n_families).colors

        # Create individual plots for each family
        for i, (eval_rewards, train_rewards, mean_eval_rewards) in enumerate(zip(self.families_eval_rewards, self.families_train_rewards, self.families_mean_eval_rewards)):
            plt.figure()
            plt.clf()
            plt.title(f'Family {i+1} Rewards - {self.step_count} steps')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            color = colors[i % len(colors)]
            plt.plot(eval_rewards, label=f'Episode Reward (current)', color=color, linestyle='-')
            plt.plot(train_rewards, label=f" Train HOF Episodes Reward (current)", color=color, linestyle=':')
            plt.plot(mean_eval_rewards, label=f"{self.plot_eval_window}-Episodes Reward (window)", color=color, linestyle='--')

            plt.legend()
            plt.grid()

            # Save the individual plot for this family
            plt.savefig(f"{self.plot_path}_family_{i+1}.png", bbox_inches="tight")
            plt.close()  # Close the figure to free memory


        # Create a sigle plot with individual subplots for each family
        num_rows = 2
        num_cols = math.ceil(self.n_families / num_rows)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(7*num_cols, 10))
        axes = axes.flatten()

        for i, (eval_rewards, train_rewards, mean_eval_rewards) in enumerate(zip(self.families_eval_rewards, self.families_train_rewards, self.families_mean_eval_rewards)):
            ax = axes[i]
            ax.set_title(f'Family {i+1} Rewards - {self.step_count} steps')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')

            color = colors[i % len(colors)]
            ax.plot(eval_rewards, label=f'Episode Reward (current)', color=color, linestyle='-')
            ax.plot(train_rewards, label=f"Train HOF Episodes Reward (current)", color=color, linestyle=':')
            ax.plot(mean_eval_rewards, label=f"{self.plot_eval_window}-Episodes Reward (window)", color=color, linestyle='--')

            ax.legend()
            ax.grid()

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{self.plot_path}_subplots.png", bbox_inches="tight")
        plt.close()

        # Create a combined plot for all average rewards
        plt.figure()
        plt.clf()
        plt.title(f'All Families Average Rewards (window {self.plot_eval_window}) - {self.step_count} steps')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        for i, mean_eval_rewards in enumerate(self.families_mean_eval_rewards):
            color = colors[i % len(colors)]
            plt.plot(mean_eval_rewards, label=f"Family {i+1}", color=color, linestyle='--')

        plt.legend()
        plt.grid()

        # Save the combined plot
        plt.savefig(f"{self.plot_path}_all_families.png", bbox_inches="tight")
        plt.close()  # Close the figure to free memory


    def plot_collected_rewards(self):
        plt.figure(1)
        plt.clf()
        plt.title(f'{self.step_count} steps - Average Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        avg_eval_rewards = np.mean(np.array(self.families_eval_rewards), axis=0)
        avg_mean_eval_rewards = np.mean(np.array(self.families_mean_eval_rewards), axis=0)

        max_eval_rewards = np.max(np.array(self.families_eval_rewards), axis=0)
        max_mean_eval_rewards = np.max(np.array(self.families_mean_eval_rewards), axis=0)

        plt.plot(avg_eval_rewards, label='Average Episode Reward', color='blue', linestyle='-')
        plt.plot(avg_mean_eval_rewards, color='blue', linestyle='--')
        plt.plot(max_eval_rewards, label='Max Episode Reward', color='red', linestyle='-')
        plt.plot(max_mean_eval_rewards, color='red', linestyle='--')

        plt.legend()
        plt.grid()

        avg_plot_path = self.plot_path + '_collected.png'
        plt.savefig(avg_plot_path, bbox_inches="tight")
        plt.close()  # Close the figure to free memory


    def start_log(self, path):
        path = path + '.txt'
        with open(path, 'w') as file:
            file.write('Starting game:\n\n')


    def write_log(self, path, agent, player, action, logits, mlogits, probs):

        path = path + '.txt'
        last_layer = list(agent.model.children())[-1]
        has_bias = hasattr(last_layer, 'bias') and last_layer.bias is not None

        with open(path, 'a') as file:
            file.write("-"*91 + "\n")
            file.write(f"{'Agent':<10} {player}\n")
            file.write(f"{'Action':<10} {action}\n")
            file.write("-"*91 + "\n")
            file.write(f"{' ':<10} {' '.join(f'{val:>8}' for val in range(self.n_actions))}\n")
            file.write(f"{'MLogits':<10} {' '.join(f'{val:>8.4f}' for val in mlogits.flatten())}\n")
            file.write(f"{'Logits':<10} {' '.join(f'{val:>8.4f}' for val in logits.flatten())}\n")
            if has_bias:
                file.write(f"{'Bias':<10} {' '.join(f'{val:>8.4f}' for val in last_layer.bias.flatten())}\n")
                differences = logits.flatten() - last_layer.bias.flatten()
                file.write(f"{'Diff':<10} {' '.join(f'{val:>8.4f}' for val in differences)}\n")
            file.write(f"{'Probs':<10} {' '.join(f'{val:>8.4f}' for val in probs.flatten())}\n")
            file.write("-"*91 + "\n\n\n")


    def compose_video(self, path, frames):
        path = path + '.mp4'
        with imageio.get_writer(path, fps=15) as writer:
            for frame in frames:
                for i in range(15):
                    writer.append_data(np.array(frame))


    def save_winner(self, filename):
        self.winner.save(filename)
        return True


    def save_winners(self, filename):
        for i in range(self.n_families):
            name = filename.replace('.pt', f'{i}.pt')
            self.family_winners[i].save(name)
        return True