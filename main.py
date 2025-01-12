import argparse
import logging

from evolutionary_trainer import *
from genetic_trainer import *

def main():

    parser = argparse.ArgumentParser(description="Select a type of training to conduct")

    # Add a subparser for the two types of training
    subparsers = parser.add_subparsers(dest="training_type", required=True, help="Type of training to perform")
    evo_parser = subparsers.add_parser('EMS', help="Train using evolutionary strategy")
    gen_parser = subparsers.add_parser('GMA', help="Train using genetic algorithm")

    # Shared Parser arguments
    parser.add_argument("--env_type", type=str, default='tictactoe_v3', help="Type of environment to train on", choices = ['tictactoe_v3', 'connect_four_v3', 'texas_holdem_no_limit_v6'])
    parser.add_argument("--n_families", type=int, default=4, help="Number of families")
    parser.add_argument("--family_size", type=int, default=25, help="Size of each family")
    parser.add_argument("--initial_std_dev", type=float, default=0.09, help="Initial standard deviation")
    parser.add_argument("--min_std_dev", type=float, default=0.01, help="Minimum standard deviation")
    parser.add_argument("--std_dev_decay", type=float, default=0.995, help="Standard deviation decay")
    parser.add_argument("--n_generations", type=int, default=600, help="Number of generations")
    parser.add_argument("--gamma", type=float, default=0.99, help="Reward decay")
    parser.add_argument("--neg_multiplier", type=float, default=1.1, help="How much to increase the reward when negative")
    parser.add_argument("--family_hof_size", type=int, default=5, help="Size of the hall of fame of each family")
    parser.add_argument("--use_action_mask", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Whether to use action mask")
    parser.add_argument("--plot_eval_freq", type=int, default=1, help="Frequency of plotting")
    parser.add_argument("--plot_eval_times", type=int, default=50, help="Number of times to evaluate for plotting")
    parser.add_argument("--plot_eval_window", type=int, default=20, help="Window size for plotting")
    parser.add_argument("--use_softmax", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Whether to use softmax")
    parser.add_argument("--plot_path", type=str, default="./reward_plot_episode.png", help="Path to save plots")
    parser.add_argument("--video_folder", type=str, default="./videos", help="Folder to save videos")
    parser.add_argument("--parallelization_type", type=str, default='no', choices=['no', 'hof', 'family'], help="at which level parallelize episode execution")
    parser.add_argument("--network_type", type=str, default="ClassicNet", choices=["ClassicNet", "DeepNet"], help="Which network to use")

    # Add arguments for evolutionary trainer
    evo_parser.add_argument("--normalize_gradient", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Consider the mean for each family for the fitness")
    evo_parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")

    # Add arguments for genetic trainer
    gen_parser.add_argument("--family_n_elites", type=int, default=5, help="Number of elites for each family")

    args = parser.parse_args()

    if args.training_type == 'GMA':
        args_dict = vars(gen_parser.parse_args())
        trainer = GeneticMultiTrainer(**args_dict)
    
    elif args.training_type == 'EMS':
        args_dict = vars(evo_parser.parse_args())
        trainer = EvolutionMultiTrainer(**args_dict)

    #train 
    trainer.train()

    #save the winner
    trainer.save_winner("./winner.pkl")


if __name__ == "__main__":
    logging.getLogger("pettingzoo").setLevel(logging.ERROR)
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    main()