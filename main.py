from evolutionary_trainer import EvolutionTrainer
from genetic_trainer import GeneticTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Select a type of training to conduct")

    # Add a subparser for the two types of training
    subparsers = parser.add_subparsers(dest="training_type", required=True, help="Type of training to perform")

    # Evolutionary trainer arguments
    evo_parser = subparsers.add_parser('evolutionary', help="Train using evolutionary strategy")

    gen_parser = subparsers.add_parser('genetic', help="Train using genetic algorithm")

    # Add arguments for evolutionary trainer
    evo_parser.add_argument("--env_type", type=str, help="Type of environment to train on", default='tictactoe_v3', choices = ['tictactoe_v3', 'connect_four_v3', 'texas_holdem_no_limit_v6'])
    evo_parser.add_argument("--population_size", type=int, help="Size of the population", default=200)
    evo_parser.add_argument("--n_generations", type=int, help="Number of generations", default=300)
    evo_parser.add_argument("--use_softmax", type=bool, help="Whether to use softmax", default=False)
    evo_parser.add_argument("--hof_size", type=int, help="Size of the hall of fame", default = 10)
    evo_parser.add_argument("--dummy_size", type=int, help="Size of the dummy population", default = 10)
    evo_parser.add_argument("--dummy_decay_freq", type=int, help="Frequency of dummy decay", default = 5)
    evo_parser.add_argument("--initial_std_dev", type=float, help="Initial standard deviation", default = 0.1)
    evo_parser.add_argument("--min_std_dev", type=float, help="Minimum standard deviation", default = 0.001)
    evo_parser.add_argument("--std_dev_decay", type=float, help="Standard deviation decay", default = 0.99)
    evo_parser.add_argument("--dissimilarity_weight", type=float, help="Dissimilarity weight", default = 0.5)
    evo_parser.add_argument("--learning_rate", type=float, help="Learning rate", default = 0.01)
    evo_parser.add_argument("--plot_eval_times", type=int, help="Number of times to evaluate for plotting", default = 1)
    evo_parser.add_argument("--plot_eval_freq", type=int, help="Frequency of plotting", default = 10)
    evo_parser.add_argument("--plot_eval_window", type=int, help="Window size for plotting", default = 20)
    evo_parser.add_argument("--use_action_mask", type=bool, help="Whether to use action mask", default = True)
    evo_parser.add_argument("--plot_path", type=str, help="Path to save plots", default = "/content/reward_plot_episode.png")
    evo_parser.add_argument("--video_folder", type=str, help="Folder to save videos", default = "/content/videos")

    # Add arguments for genetic trainer
    
    gen_parser.add_argument("--env_type", type=str, help="Type of environment to train on", default = 'tictactoe_v3', choices = ['tictactoe_v3', 'connect_four_v3', 'texas_holdem_no_limit_v6'])
    gen_parser.add_argument("--population_size", type=int, help="Size of the population", default = 200)
    gen_parser.add_argument("--n_generations", type=int, help="Number of generations", default = 300)
    gen_parser.add_argument("--use_softmax", type=bool, help="Whether to use softmax", default = False)
    gen_parser.add_argument("--hof_size", type=int, help="Size of the hall of fame", default = 10)
    gen_parser.add_argument("--dummy_size", type=int, help="Size of the dummy population", default = 10)
    gen_parser.add_argument("--dummy_decay_freq", type=int, help="Frequency of dummy decay", default = 5)
    gen_parser.add_argument("--n_elites", type=int, help="Number of elites", default = 10)
    gen_parser.add_argument("--initial_std_dev", type=float, help="Initial standard deviation", default = 0.1)
    gen_parser.add_argument("--min_std_dev", type=float, help="Minimum standard deviation", default = 0.001)
    gen_parser.add_argument("--std_dev_decay", type=float, help="Standard deviation decay", default = 0.99)
    gen_parser.add_argument("--dissimilarity_weight", type=float, help="Dissimilarity weight", default = 0.5)
    gen_parser.add_argument("--plot_eval_times", type=int, help="Number of times to evaluate for plotting", default = 1)
    gen_parser.add_argument("--plot_eval_freq", type=int, help="Frequency of plotting", default = 10)
    gen_parser.add_argument("--plot_eval_window", type=int, help="Window size for plotting", default = 20)
    gen_parser.add_argument("--use_action_mask", type=bool, help="Whether to use action mask", default = True)
    gen_parser.add_argument("--plot_path", type=str, help="Path to save plots", default = "/content/reward_plot_episode.png")
    gen_parser.add_argument("--video_folder", type=str, help="Folder to save videos", default = "/content/videos")

    args = parser.parse_args()

    if args.training_type == 'genetic':
        trainer = GeneticTrainer(args.env_type, args.population_size, args.n_generations,
                                 args.use_softmax, args.hof_size, args.dummy_size, args.dummy_decay_freq,
                                 args.n_elites, args.initial_std_dev, args.min_std_dev, args.std_dev_decay,
                                 args.dissimilarity_weight, args.plot_eval_times, args.plot_eval_freq,
                                 args.plot_eval_window, args.use_action_mask)
    
    elif args.training_type == 'evolutionary':
        trainer = EvolutionTrainer(args.env_type, args.population_size, args.n_generations,
                                   args.use_softmax, args.hof_size, args.dummy_size, args.dummy_decay_freq,
                                   args.initial_std_dev, args.min_std_dev, args.std_dev_decay,
                                   args.dissimilarity_weight, args.learning_rate, args.plot_eval_times,
                                   args.plot_eval_freq, args.plot_eval_window, args.use_action_mask)
    #train 
    trainer.train()

    #play against the winner
    trainer.play_against(trainer.winner, show_scores=True)

    #save the winner
    trainer.save_winner("./winner.pkl")

if __name__ == "__main__":
    main()