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

    # Shared arguments for both training types
    for subparser in (evo_parser, gen_parser):
        subparser.add_argument("--env_type", type=str, default='tictactoe_v3', 
                               help="Type of environment to train on", 
                               choices=['tictactoe_v3', 'connect_four_v3', 'texas_holdem_no_limit_v6'])
        subparser.add_argument("--n_families", type=int, default=4, help="Number of families")
        subparser.add_argument("--family_size", type=int, default=25, help="Size of each family")
        subparser.add_argument("--initial_std_dev", type=float, default=0.09, help="Initial standard deviation")
        subparser.add_argument("--min_std_dev", type=float, default=0.01, help="Minimum standard deviation")
        subparser.add_argument("--std_dev_decay", type=float, default=0.995, help="Standard deviation decay")
        subparser.add_argument("--n_generations", type=int, default=600, help="Number of generations")
        subparser.add_argument("--gamma", type=float, default=0.99, help="Reward decay")
        subparser.add_argument("--neg_multiplier", type=float, default=1.1, help="How much to increase the reward when negative")
        subparser.add_argument("--family_hof_size", type=int, default=5, help="Size of the hall of fame of each family")
        subparser.add_argument("--use_action_mask", type=bool, default=True, action=argparse.BooleanOptionalAction, 
                               help="Whether to use action mask")
        subparser.add_argument("--plot_eval_freq", type=int, default=1, help="Frequency of plotting")
        subparser.add_argument("--plot_eval_times", type=int, default=50, help="Number of times to evaluate for plotting")
        subparser.add_argument("--plot_eval_window", type=int, default=20, help="Window size for plotting")
        subparser.add_argument("--use_softmax", type=bool, default=False, action=argparse.BooleanOptionalAction, 
                               help="Whether to use softmax")
        subparser.add_argument("--plot_path", type=str, default="./reward_plot_episode.png", help="Path to save plots")
        subparser.add_argument("--video_folder", type=str, default="./videos", help="Folder to save videos")
        subparser.add_argument("--parallelization_type", type=str, default='no', 
                               choices=['no', 'hof', 'family'], 
                               help="At which level to parallelize episode execution")
        subparser.add_argument("--network_type", type=str, default="ClassicNet", 
                               choices=["ClassicNet", "DeepNet"], 
                               help="Which network to use")

    # Specific arguments for evolutionary trainer
    evo_parser.add_argument("--normalize_gradient", type=bool, default=False, action=argparse.BooleanOptionalAction, 
                             help="Consider the mean for each family for the fitness")
    evo_parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")

    # Specific arguments for genetic trainer
    gen_parser.add_argument("--family_n_elites", type=int, default=5, help="Number of elites for each family")

    args = parser.parse_args()
    training_type = args.training_type
    input_args = vars(args)
    input_args.pop("training_type")

    if training_type == 'GMA':
        trainer = GeneticMultiTrainer(**input_args)
    elif training_type == 'EMS':
        trainer = EvolutionMultiTrainer(**input_args)

    # Create folders
    os.makedirs(input_args["video_folder"], exist_ok=True)

    # Train the model
    trainer.train()

    # Save the winner
    trainer.save_winner("./winner.pkl")


if __name__ == "__main__":
    logging.getLogger("pettingzoo").setLevel(logging.ERROR)
    logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
    main()
