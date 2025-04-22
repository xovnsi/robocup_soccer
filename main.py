import argparse
import multiprocessing
from game_controller import GameController

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RoboCup Soccer Simulator Client')
    parser.add_argument('--players_count', type=int, default=3,
                      help='Number of field players in each team (excluding goalie) (default: 3)')
    parser.add_argument('--game_duration', type=int, default=300,
                      help='Duration of the game in seconds (default: 300)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Print welcome message
    print(f"Starting RoboCup simulation with {args.players_count} field players per team plus goalies...")
    print(f"Game will run for {args.game_duration} seconds.")
    
    # Ensure proper multiprocessing startup on some platforms (like Windows)
    multiprocessing.freeze_support()
    
    # Create and run the game
    game = GameController(players_count=args.players_count, game_duration=args.game_duration)
    game.run_game()

if __name__ == "__main__":
    main()
