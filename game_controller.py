import time
import multiprocessing
from threading import Thread
from utils import create_rcss_agent, calculate_formations
from player_behaviors import active_player_behavior, goalie_behavior, ensure_activity

class GameController:
    def __init__(self, players_count=3, game_duration=300):
        self.players_count = players_count
        self.game_duration = game_duration
        self.agents = []
        self.left_agents = []
        self.right_agents = []
        self.threads = []
        
    def setup_teams(self):
        """Create and position all players for both teams."""
        # Field positions
        goal_x_left = -52
        goal_x_right = 52
        goal_y = 0
        
        # Create goalies
        print("Creating goalies...")
        proc, cmd_q, resp_q = create_rcss_agent("left", 1, is_goalie=True)
        if proc:
            self.agents.append((proc, cmd_q, resp_q, "left_goalie"))
            self.left_agents.append((proc, cmd_q, resp_q, "left_goalie"))
            cmd_q.put({"action": "move", "position": (goal_x_left + 2, goal_y)})
            response = resp_q.get(timeout=1)
            print(f"left_goalie: {response['message']}")
        
        proc, cmd_q, resp_q = create_rcss_agent("right", 1, is_goalie=True)
        if proc:
            self.agents.append((proc, cmd_q, resp_q, "right_goalie"))
            self.right_agents.append((proc, cmd_q, resp_q, "right_goalie"))
            cmd_q.put({"action": "move", "position": (goal_x_right - 2, goal_y)})
            response = resp_q.get(timeout=1)
            print(f"right_goalie: {response['message']}")
        
        # Create field players for left team
        print(f"Creating {self.players_count} field players for left team...")
        for i in range(self.players_count):
            proc, cmd_q, resp_q = create_rcss_agent("left", i+2)
            if proc:
                self.agents.append((proc, cmd_q, resp_q, f"left_{i+2}"))
                self.left_agents.append((proc, cmd_q, resp_q, f"left_{i+2}"))
        
        # Create field players for right team
        print(f"Creating {self.players_count} field players for right team...")
        for i in range(self.players_count):
            proc, cmd_q, resp_q = create_rcss_agent("right", i+2)
            if proc:
                self.agents.append((proc, cmd_q, resp_q, f"right_{i+2}"))
                self.right_agents.append((proc, cmd_q, resp_q, f"right_{i+2}"))
        
        # Calculate formation positions
        left_positions, right_positions = calculate_formations(self.players_count)
        
        # Position left team players
        print("Positioning left team players...")
        for i, (_, cmd_q, resp_q, name) in enumerate(self.left_agents[1:], 0):  # Skip goalie
            if i < len(left_positions):
                cmd_q.put({"action": "move", "position": left_positions[i]})
                response = resp_q.get(timeout=1)
                print(f"{name}: {response['message']}")
        
        # Position right team players
        print("Positioning right team players...")
        for i, (_, cmd_q, resp_q, name) in enumerate(self.right_agents[1:], 0):  # Skip goalie
            if i < len(right_positions):
                cmd_q.put({"action": "move", "position": right_positions[i]})
                response = resp_q.get(timeout=1)
                print(f"{name}: {response['message']}")
        
        # Have goalies face the field
        print("Orienting goalies...")
        _, cmd_q, resp_q, name = self.left_agents[0]  # Left goalie
        cmd_q.put({"action": "turn", "moment": 0})  # Face right (field)
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        
        _, cmd_q, resp_q, name = self.right_agents[0]  # Right goalie
        cmd_q.put({"action": "turn", "moment": 180})  # Face left (field)
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
    
    def start_kickoff(self):
        """Move a player to center for kickoff and initialize ball movement."""
        print("Moving a player to center for kickoff...")
        if len(self.left_agents) > 1:
            _, cmd_q, resp_q, name = self.left_agents[1]  # First field player of left team
            print(f"Moving {name} to center...")
            cmd_q.put({"action": "move", "position": (0, 1)})
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")

            # turn to face the ball (upwards)
            cmd_q.put({"action": "turn", "moment": -90})
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")
            
            # Make sure this player kicks the ball
            time.sleep(1)
            cmd_q.put({"action": "kick", "power": 50, "direction": 90})
            print(f"{name} kicking the ball...")
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")
            
            # Have this player kick again several times to ensure ball movement
            for _ in range(3):
                time.sleep(0.5)
                cmd_q.put({"action": "kick", "power": 100, "direction": 0})
                response = resp_q.get(timeout=1)
    
    def start_player_threads(self):
        """Start all player behavior threads."""
        print("Starting gameplay with enhanced movement...")
        
        # Start field player threads
        for i, agent in enumerate(self.left_agents[1:], 0):
            role = i % 3  # 0: defender, 1: midfielder, 2: forward
            thread = Thread(target=active_player_behavior, args=(agent, True, role))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        for i, agent in enumerate(self.right_agents[1:], 0):
            role = i % 3
            thread = Thread(target=active_player_behavior, args=(agent, False, role))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        # Start goalie threads
        left_goalie_thread = Thread(target=goalie_behavior, args=(self.left_agents[0], True))
        left_goalie_thread.daemon = True
        left_goalie_thread.start()
        self.threads.append(left_goalie_thread)
        
        right_goalie_thread = Thread(target=goalie_behavior, args=(self.right_agents[0], False))
        right_goalie_thread.daemon = True
        right_goalie_thread.start()
        self.threads.append(right_goalie_thread)
        
        # Start activity enforcement thread
        activity_thread = Thread(target=ensure_activity, args=(self.left_agents, self.right_agents))
        activity_thread.daemon = True
        activity_thread.start()
        self.threads.append(activity_thread)
    
    def run_game(self):
        """Run the complete game."""
        try:
            # Setup teams and initial positions
            self.setup_teams()
            
            # Add a small delay before kickoff
            time.sleep(2)
            
            # Perform kickoff
            self.start_kickoff()
            
            # Start player behavior threads
            self.start_player_threads()
            
            # Game timer with periodic reporting
            print(f"Game started! Will run for {self.game_duration} seconds...")
            start_time = time.time()
            
            while time.time() - start_time < self.game_duration:
                remaining = int(self.game_duration - (time.time() - start_time))
                if remaining % 30 == 0 and remaining > 0:  # Report every 30 seconds
                    print(f"Game time remaining: {remaining} seconds")
                    
                    # Force a central player to dash toward ball
                    if len(self.left_agents) > 2:  # Use a midfielder
                        _, cmd_q, resp_q, name = self.left_agents[2]  # Midfielder
                        cmd_q.put({"action": "dash", "power": 100, "direction": 0})
                        try:
                            resp_q.get(timeout=0.5)
                        except multiprocessing.queues.Empty:
                            pass
                        print(f"{name} forced to dash toward center")
                
                time.sleep(1)
            
            print("Game time is up! Ending the match...")
        
        except KeyboardInterrupt:
            print("\nGame interrupted by user!")
        
        finally:
            # Clean up by sending exit command to all agents
            self.cleanup()
    
    def cleanup(self):
        """Clean up and terminate all processes."""
        print("Shutting down all agents...")
        for proc, cmd_q, _, name in self.agents:
            cmd_q.put({"action": "exit"})
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
        
        print("Game ended.")