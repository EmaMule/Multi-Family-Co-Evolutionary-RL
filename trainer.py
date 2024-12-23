from pettingzoo.classic import tictactoe_v3, connect_four_v3, texas_holdem_no_limit_v6
import imageio
from tqdm import tqdm
from collections import deque

import matplotlib.pyplot as plt
from IPython.display import clear_output

from abc import ABC, abstractmethod

class Trainer(ABC):

    def __init__(self, env_type, render_mode, population_size, n_generations,
                 use_softmax, hof_size, dummy_size, dummy_decay_freq, initial_std_dev, min_std_dev,
                 std_dev_decay, dissimilarity_weight, plot_eval_times, plot_eval_freq,
                 plot_eval_window, use_action_mask):

        assert env_type in [tictactoe_v3, connect_four_v3, texas_holdem_no_limit_v6]
        assert render_mode in ["rgb_array", "human"]

        # environment parameters
        self.env_type = env_type
        self.render_mode = render_mode

        # training parameters
        self.use_softmax = use_softmax
        self.population_size = population_size
        self.hof_size = hof_size # total dimension of HOF
        self.init_dummy_size = dummy_size # number of dummies in HOF
        self.dummy_decay_freq = dummy_decay_freq # steps to remove one dummy
        self.n_generations = n_generations
        self.initial_std_dev = initial_std_dev
        self.min_std_dev = min_std_dev
        self.std_dev_decay = std_dev_decay
        self.dissimilarity_weight = dissimilarity_weight
        self.use_action_mask = use_action_mask

        # parameters depending on the environment
        if self.env_type == tictactoe_v3:
            self.input_shape = [3,3,2]
            self.n_actions = 9
            self.players = ['player_1', 'player_2']
            self.transform_obs = lambda x: x.permute(2, 0, 1).unsqueeze(0)
        elif self.env_type == connect_four_v3:
            self.input_shape = [6,7,2]
            self.n_actions = 7
            self.players = ['player_0', 'player_1']
            self.transform_obs = lambda x: x.permute(2, 0, 1).unsqueeze(0)
        else:
            self.input_shape = [54]
            self.n_actions = 5
            self.players = ['player_0', 'player_1']
            self.transform_obs = lambda x: x.unsqueeze(0)

        # plot parameters
        self.dummy = DummyAgent(self.n_actions)
        self.plot_eval_window = plot_eval_window
        self.plot_eval_times = plot_eval_times
        self.plot_eval_freq = plot_eval_freq
        self.plot_path = "/content/reward_plot_episode.png"

        # video parameters
        self.video_folder = "/content/videos"

        # initialize variables
        self.dummy_size = self.init_dummy_size
        self.eval_rewards = []
        self.mean_eval_rewards = []
        self.std_eval_rewards = []
        self.step_count = 0
        self.std_dev = self.initial_std_dev


    def play_game(self, agent1, agent2, save_video = False):

        env = self.env_type.env(render_mode=self.render_mode)
        env.reset()

        frames = []
        path = self.video_folder + "/epoch_"+str(self.step_count) + '.txt'

        if save_video:
            with open(path, 'w') as file:
                file.write('Starting game:\n\n')

        total_reward_1 = 0
        total_reward_2 = 0

        for player in env.agent_iter():  # AEC mode!

            observation, reward, termination, truncation, _ = env.last()
            done = termination or truncation

            if player == self.players[0]:
                agent = agent1
                total_reward_1 += reward
            else:
                agent = agent2
                total_reward_2 += reward

            if done:
                action = None

            else:

                mask = torch.tensor(observation["action_mask"], dtype=torch.uint8)
                obs = torch.tensor(observation['observation'], dtype=torch.float32)
                obs = self.transform_obs(obs)

                # an agent can do the wrong action in 'training'
                if not self.use_action_mask and agent.mode == 'training':
                    mask = torch.ones_like(mask)

                action, logits, probs = agent.choose_action(obs, mask)

            env.step(action)

            # save the rendered frame for the video
            if save_video:
                frame = env.render()
                frames.append(frame)

                with open(path, 'a') as file:
                    file.write(f'Agent: {player}\n')
                    file.write(f'Action: {action}\n')
                    file.write(f'Logits: {", ".join(f"{val:.4f}" for val in logits.flatten())}\n')
                    file.write(f'Probs: {", ".join(f"{val:.4f}" for val in probs.flatten())}\n\n')

        env.close()

        if save_video:
            self.compose_video(frames)

        return total_reward_1, total_reward_2


    # evaluate the agent by playing against the evaluator
    def evaluate_agent(self, agent, evaluator, times = 1):
        total_reward = 0
        agent.mode = 'training'
        evaluator.mode = 'evaluating'
        for i in range(times):
            total_reward += self.play_game(evaluator, agent)[1]
            total_reward += self.play_game(agent, evaluator)[0]
        return total_reward / times


    def record_play(self, agent1, agent2):
        agent1.mode = 'deploying'
        agent2.mode = 'deploying'
        rewards = self.play_game(agent1, agent2, save_video = True)
        return rewards


    # make the human play against the agent
    def play_against(self, agent, show_scores):

        env = self.env_type.env(render_mode="rgb_array")
        env.reset()

        agent.mode = 'deploying'

        total_reward_user = 0
        total_reward_agent = 0
        action = None
        logits = None
        probs = None

        for agent_name in env.agent_iter():  # AEC mode!

            observation, reward, termination, truncation, _ = env.last()
            done = termination or truncation

            # visualize current state
            frame = env.render()
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(frame)
            ax.axis('off')
            plt.show()

            # human's turn
            if agent_name == self.players[0]:
                total_reward_user += reward

            # agent's turn
            else:
                total_reward_agent += reward

            if done:
                action = None

            else:

                mask = torch.tensor(observation["action_mask"], dtype=torch.uint8)
                obs = torch.tensor(observation['observation'], dtype=torch.float32)
                obs = self.transform_obs(obs)

                # human's turn
                if agent_name == self.players[0]:

                    # log of agent
                    if show_scores:
                        print("---")
                        print("Agent's Action:\n", action)
                        print("Agent's Logits:\n", logits)
                        print("Agent's Probabilities:\n", probs)
                        print("---\n\n")

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
                    action, logits, probs = agent.choose_action(obs, mask)

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
        print("You won!" if total_reward_user > total_reward_agent else "You lose!")


    @abstractmethod
    def train_step(self): # must be implemented by every class
        pass


    def train(self):

        # initialize variables
        self.dummy_size = self.init_dummy_size
        self.eval_rewards = []
        self.mean_eval_rewards = []
        self.std_eval_rewards = []
        self.step_count = 0
        self.std_dev = self.initial_std_dev

        # training loop
        for t in tqdm(range(self.n_generations)):

            # train
            self.train_step()

            # evaluation
            reward = self.evaluate_agent(self.winner, self.dummy, self.plot_eval_times)
            self.update_metrics(reward)
            if self.step_count % self.plot_eval_freq == 0 and self.step_count != 0:
                self.plot_rewards()

            if self.step_count % self.dummy_decay_freq == 0 and self.step_count != 0 and self.dummy_size > 0:
                self.dummy_size -= 1

            self.step_count+=1
            self.std_dev = max(self.min_std_dev, self.std_dev * self.std_dev_decay)


    def update_metrics(self, reward):
        self.eval_rewards.append(reward)
        n_elems = min(len(self.eval_rewards), self.plot_eval_window)
        reward_window = self.eval_rewards[-n_elems:]
        self.mean_eval_rewards.append(np.mean(reward_window))
        self.std_eval_rewards.append(np.std(reward_window))


    def plot_rewards(self):
        plt.figure(1)
        plt.clf()
        plt.title(f'{self.step_count} steps, reward {self.mean_eval_rewards[-1]:.2f}±{self.std_eval_rewards[-1]:.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.eval_rewards, label='Episode Reward')
        plt.plot(self.mean_eval_rewards, label=f"{self.plot_eval_window}-Episodes Reward Average", linestyle='--')
        plt.legend()
        plt.grid()
        plt.savefig(self.plot_path, bbox_inches="tight")


    def compose_video(self, frames):
        path = self.video_folder + "/epoch_"+str(self.step_count) + '.mp4'
        with imageio.get_writer(path, fps=1) as writer:
            for frame in frames:
                writer.append_data(np.array(frame))