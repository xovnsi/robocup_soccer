import multiprocessing
import time
import socket
import re
import random
import argparse  # Added for command line argument parsing

class RCSSClient:
    def __init__(self, team_name="TeamPython", player_num=1, is_goalie=False, server_host="localhost", server_port=6000):
        self.team_name = team_name
        self.player_num = player_num
        self.is_goalie = is_goalie
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.is_connected = False
        self.position = (0, 0)
        
    def connect(self):
        """Połączenie z serwerem RCSS."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(0.5)
            
            # Add goalie designation if this is a goalie
            init_msg = f"(init {self.team_name} (version 19){' (goalie)' if self.is_goalie else ''})"
            self.socket.sendto(init_msg.encode(), (self.server_host, self.server_port))
            
            data, addr = self.socket.recvfrom(4096)
            response = data.decode()
            
            if "(init" in response:
                print(f"[{self.team_name}] {'Goalie' if self.is_goalie else 'Player'} połączony z serwerem.")
                self.is_connected = True
                self.player_num = int(response.split()[2])
                self.server_port = addr[1]
                return True
            else:
                print(f"[{self.team_name}] Błąd połączenia: {response}")
                return False
                
        except Exception as e:
            print(f"[{self.team_name}] Błąd: {str(e)}")
            return False
    
    def move_to_position(self, x, y):
        """Przemieszczenie agenta na wskazaną pozycję."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        move_msg = f"(move {x} {y})"
        self.socket.sendto(move_msg.encode(), (self.server_host, self.server_port))
        self.position = (x, y)
        return f"Przemieszczono na pozycję ({x}, {y})"
    
    def dash(self, power, direction=0):
        """Wykonanie ruchu do przodu z określoną siłą."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        dash_msg = f"(dash {power} {direction})"
        self.socket.sendto(dash_msg.encode(), (self.server_host, self.server_port))
        return f"Wykonano dash z mocą {power} i kierunkiem {direction}"
    
    def turn(self, moment):
        """Obrót agenta o zadany kąt."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        turn_msg = f"(turn {moment})"
        self.socket.sendto(turn_msg.encode(), (self.server_host, self.server_port))
        return f"Wykonano obrót o {moment}"

    def catch(self, direction):
        """Próba złapania piłki przez bramkarza."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
        if not self.is_goalie:
            return "Tylko bramkarz może łapać piłkę"
            
        catch_msg = f"(catch {direction})"
        self.socket.sendto(catch_msg.encode(), (self.server_host, self.server_port))
        return f"Próba złapania piłki w kierunku {direction}"
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.is_connected = False
            print(f"[{self.team_name}] Rozłączono.")
    
    def run(self, cmd_queue, resp_queue):
        """Główna pętla agenta."""
        if not self.connect():
            resp_queue.put({"status": "error", "message": "Nie udało się połączyć z serwerem"})
            return
        
        resp_queue.put({"status": "connected", "player_num": self.player_num, "is_goalie": self.is_goalie})
        
        try:
            while True:
                # Sprawdź czy są nowe komendy
                try:
                    cmd = cmd_queue.get(block=False)
                    if cmd["action"] == "move":
                        x, y = cmd["position"]
                        result = self.move_to_position(x, y)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "dash":
                        power = cmd.get("power", 100)
                        direction = cmd.get("direction", 0)
                        result = self.dash(power, direction)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "turn":
                        moment = cmd.get("moment", 30)
                        result = self.turn(moment)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "catch" and self.is_goalie:
                        direction = cmd.get("direction", 0)
                        result = self.catch(direction)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "exit":
                        break
                except multiprocessing.queues.Empty:
                    pass  # Brak nowych komend
                
                # Tutaj można dodać odbieranie i przetwarzanie komunikatów z serwera
                
                time.sleep(0.01)  # Krótkie opóźnienie
                
        except Exception as e:
            resp_queue.put({"status": "error", "message": str(e)})
        finally:
            self.disconnect()

def create_rcss_agent(team, player_num, is_goalie=False):
    """Tworzy i uruchamia agenta RCSS."""
    cmd_queue = multiprocessing.Queue()
    resp_queue = multiprocessing.Queue()
    
    client = RCSSClient(team_name=team, player_num=player_num, is_goalie=is_goalie)
    process = multiprocessing.Process(target=client.run, args=(cmd_queue, resp_queue))
    process.start()
    
    # Poczekaj na połączenie
    response = resp_queue.get(timeout=5)
    if response["status"] == "error":
        print(f"Błąd: {response['message']}")
        process.terminate()
        return None, None, None
    
    player_type = "Goalie" if response.get("is_goalie", False) else "Player"
    print(f"Agent {team} #{response['player_num']} ({player_type}) połączony")
    
    return process, cmd_queue, resp_queue

def run_rcss_example(players_count=1):
    """
    Uruchamia symulację RoboCup z określoną liczbą agentów dla każdej drużyny.
    
    Args:
        players_count (int): Liczba agentów w każdej drużynie (nie licząc bramkarza)
    """
    agents = []
    
    # Field dimensions for positioning
    # The RoboCup field is typically about 105x68 meters
    # In simulator coordinates, this might be roughly -52.5 to 52.5 on x-axis
    # and -34 to 34 on y-axis
    field_length = 105  # Standard field length
    field_width = 68    # Standard field width
    goal_x_left = -52   # Left goal x position (using integer)
    goal_x_right = 52   # Right goal x position (using integer)
    goal_y = 0          # Goal y position (center)
    
    # Create and position the left team goalie
    proc, cmd_q, resp_q = create_rcss_agent("left", 1, is_goalie=True)
    if proc:
        agents.append((proc, cmd_q, resp_q, "left_goalie"))
        # Position the goalie at the left goal
        cmd_q.put({"action": "move", "position": (goal_x_left + 2, goal_y)})
        response = resp_q.get(timeout=1)
        print(f"left_goalie: {response['message']}")
    
    # Create and position the right team goalie
    proc, cmd_q, resp_q = create_rcss_agent("right", 1, is_goalie=True)
    if proc:
        agents.append((proc, cmd_q, resp_q, "right_goalie"))
        # Position the goalie at the right goal
        cmd_q.put({"action": "move", "position": (goal_x_right - 2, goal_y)})
        response = resp_q.get(timeout=1)
        print(f"right_goalie: {response['message']}")
    
    # Stwórz agentów lewej drużyny (nie licząc bramkarza)
    for i in range(players_count):
        proc, cmd_q, resp_q = create_rcss_agent("left", i+2)  # Start from 2 because 1 is goalie
        if proc:
            agents.append((proc, cmd_q, resp_q, f"left_{i+2}"))
    
    # Stwórz agentów prawej drużyny (nie licząc bramkarza)
    for i in range(players_count):
        proc, cmd_q, resp_q = create_rcss_agent("right", i+2)  # Start from 2 because 1 is goalie
        if proc:
            agents.append((proc, cmd_q, resp_q, f"right_{i+2}"))
    
    print(f"Utworzono {len(agents)} agentów: {players_count + 1} w drużynie lewej i {players_count + 1} w drużynie prawej (w tym bramkarze).")
    
    # Calculate field position ranges and ensure they are integers
    left_field_x_min = int(goal_x_left/2)
    left_field_x_max = 0
    right_field_x_min = 0
    right_field_x_max = int(goal_x_right/2)
    field_y_min = int(-field_width/3)  
    field_y_max = int(field_width/3)
    
    # Prepare positions for field players
    left_positions = []
    for _ in range(players_count):
        x = random.randint(left_field_x_min, left_field_x_max)
        y = random.randint(field_y_min, field_y_max)
        left_positions.append((x, y))
    
    right_positions = []
    for _ in range(players_count):
        x = random.randint(right_field_x_min, right_field_x_max)
        y = random.randint(field_y_min, field_y_max)
        right_positions.append((x, y))
    
    # Combine all positions
    positions = left_positions + right_positions
    
    # Position field players (starting from index 2 as goalies are at indices 0 and 1)
    for i, (_, cmd_q, resp_q, name) in enumerate(agents[2:], 0):
        if i < len(positions):
            cmd_q.put({"action": "move", "position": positions[i]})
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")
    
    # Przykładowe sterowanie agentami
    time.sleep(5)
    
    # Have goalies turn to face the field
    if len(agents) >= 2:
        # Left goalie turns right
        _, cmd_q, resp_q, name = agents[0]  # Left goalie
        cmd_q.put({"action": "turn", "moment": 0})  # Turn to face right (field)
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        
        # Right goalie turns left
        _, cmd_q, resp_q, name = agents[1]  # Right goalie
        cmd_q.put({"action": "turn", "moment": 180})  # Turn to face left (field)
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
    
    # Jeśli są inni gracze, niech pierwszy gracz z pola się porusza
    if len(agents) > 2:
        _, cmd_q, resp_q, name = agents[2]  # First field player
        cmd_q.put({"action": "dash", "power": 100})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        time.sleep(0.2)
        cmd_q.put({"action": "dash", "power": 100})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        time.sleep(0.2)
        cmd_q.put({"action": "dash", "power": 100})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        time.sleep(0.2)
    
    # Możesz odkomentować ten kod, aby zamknąć wszystkie procesy po określonym czasie
    """
    time.sleep(15)  # Daj czas na obserwację
    for proc, cmd_q, _, _ in agents:
        cmd_q.put({"action": "exit"})
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
    """

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RoboCup Soccer Simulator Client')
    parser.add_argument('--players_count', type=int, default=1,
                      help='Number of field players in each team (excluding goalie) (default: 1)')
    return parser.parse_args()

def conduct_game(players_count=3, game_duration=300):
    """
    Conduct a full RoboCup soccer game with enhanced ball interaction.
    
    Args:
        players_count (int): Number of field players per team (excluding goalie)
        game_duration (int): Duration of the game in seconds
    """
    import time
    import random
    import multiprocessing
    from threading import Thread
    
    # Keep track of all agents
    agents = []
    left_agents = []
    right_agents = []
    
    # Field dimensions and positions
    field_length = 105
    field_width = 68
    goal_x_left = -52
    goal_x_right = 52
    goal_y = 0
    
    # Create goalies
    print("Creating goalies...")
    proc, cmd_q, resp_q = create_rcss_agent("left", 1, is_goalie=True)
    if proc:
        agents.append((proc, cmd_q, resp_q, "left_goalie"))
        left_agents.append((proc, cmd_q, resp_q, "left_goalie"))
        cmd_q.put({"action": "move", "position": (goal_x_left + 2, goal_y)})
        response = resp_q.get(timeout=1)
        print(f"left_goalie: {response['message']}")
    
    proc, cmd_q, resp_q = create_rcss_agent("right", 1, is_goalie=True)
    if proc:
        agents.append((proc, cmd_q, resp_q, "right_goalie"))
        right_agents.append((proc, cmd_q, resp_q, "right_goalie"))
        cmd_q.put({"action": "move", "position": (goal_x_right - 2, goal_y)})
        response = resp_q.get(timeout=1)
        print(f"right_goalie: {response['message']}")
    
    # Create field players for left team
    print(f"Creating {players_count} field players for left team...")
    for i in range(players_count):
        proc, cmd_q, resp_q = create_rcss_agent("left", i+2)
        if proc:
            agents.append((proc, cmd_q, resp_q, f"left_{i+2}"))
            left_agents.append((proc, cmd_q, resp_q, f"left_{i+2}"))
    
    # Create field players for right team
    print(f"Creating {players_count} field players for right team...")
    for i in range(players_count):
        proc, cmd_q, resp_q = create_rcss_agent("right", i+2)
        if proc:
            agents.append((proc, cmd_q, resp_q, f"right_{i+2}"))
            right_agents.append((proc, cmd_q, resp_q, f"right_{i+2}"))
    
    # Calculate positions for the initial formation
    # For left team: defenders, midfielders, forwards
    left_defender_x = goal_x_left + 10
    left_midfielder_x = -15
    left_forward_x = -5
    
    # For right team: defenders, midfielders, forwards
    right_defender_x = goal_x_right - 10
    right_midfielder_x = 15
    right_forward_x = 5
    
    # Position players according to formation based on their index
    # Initial positions for the left team
    left_positions = []
    field_y_values = [-15, 0, 15]  # Left, center, right positions
    
    # Simple formation: defenders, midfielders, forwards
    for i in range(players_count):
        if i % 3 == 0:  # Defenders
            left_positions.append((left_defender_x, field_y_values[i % len(field_y_values)]))
        elif i % 3 == 1:  # Midfielders
            left_positions.append((left_midfielder_x, field_y_values[i % len(field_y_values)]))
        else:  # Forwards
            left_positions.append((left_forward_x, field_y_values[i % len(field_y_values)]))
    
    # Initial positions for the right team
    right_positions = []
    for i in range(players_count):
        if i % 3 == 0:  # Defenders
            right_positions.append((right_defender_x, field_y_values[i % len(field_y_values)]))
        elif i % 3 == 1:  # Midfielders
            right_positions.append((right_midfielder_x, field_y_values[i % len(field_y_values)]))
        else:  # Forwards
            right_positions.append((right_forward_x, field_y_values[i % len(field_y_values)]))
    
    # Position left team players
    print("Positioning left team players...")
    for i, (_, cmd_q, resp_q, name) in enumerate(left_agents[1:], 0):  # Skip goalie
        if i < len(left_positions):
            cmd_q.put({"action": "move", "position": left_positions[i]})
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")
    
    # Position right team players
    print("Positioning right team players...")
    for i, (_, cmd_q, resp_q, name) in enumerate(right_agents[1:], 0):  # Skip goalie
        if i < len(right_positions):
            cmd_q.put({"action": "move", "position": right_positions[i]})
            response = resp_q.get(timeout=1)
            print(f"{name}: {response['message']}")
    
    # Have goalies face the field
    print("Orienting goalies...")
    _, cmd_q, resp_q, name = left_agents[0]  # Left goalie
    cmd_q.put({"action": "turn", "moment": 0})  # Face right (field)
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    
    _, cmd_q, resp_q, name = right_agents[0]  # Right goalie
    cmd_q.put({"action": "turn", "moment": 180})  # Face left (field)
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    
    # Move one player to the center to kick off
    print("Moving a player to center for kickoff...")
    if len(left_agents) > 1:
        _, cmd_q, resp_q, name = left_agents[1]  # First field player of left team
        cmd_q.put({"action": "move", "position": (0, 0)})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        
        # Make sure this player kicks the ball
        time.sleep(1)
        cmd_q.put({"action": "dash", "power": 100, "direction": 0})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
        
        # Have this player kick again several times to ensure ball movement
        for _ in range(3):
            time.sleep(0.5)
            cmd_q.put({"action": "dash", "power": 100, "direction": 0})
            response = resp_q.get(timeout=1)
    
    # Add a small delay to let the game start properly
    time.sleep(2)
    
    # Function to make players actively move and kick
    def active_player_behavior(agent, is_left_team, role):
        """Execute more active player behavior to ensure movement."""
        _, cmd_q, resp_q, name = agent
        
        # Direction based on team
        base_direction = 0 if is_left_team else 180
        
        while True:
            try:
                # More active behaviors to ensure movement
                action_weights = {"dash": 0.7, "turn": 0.3}  # Heavily favor dashing
                action = random.choices(list(action_weights.keys()), 
                                       weights=list(action_weights.values()))[0]
                
                if action == "dash":
                    # Much higher power values to ensure movement
                    if role == 0:  # Defender
                        power = random.uniform(80, 100)
                    elif role == 1:  # Midfielder
                        power = random.uniform(90, 100)
                    else:  # Forward
                        power = random.uniform(95, 100)
                    
                    # Slightly vary direction but mostly forward/backward
                    direction_variation = random.uniform(-20, 20)
                    direction = (base_direction + direction_variation) % 360
                    
                    cmd_q.put({"action": "dash", "power": power, "direction": direction})
                    
                elif action == "turn":
                    # More significant turns to change direction
                    moment = random.uniform(-90, 90)
                    cmd_q.put({"action": "turn", "moment": moment})
                
                # Process response
                try:
                    response = resp_q.get(timeout=0.5)
                except multiprocessing.queues.Empty:
                    pass
                
                # Always dash immediately after turning to ensure movement
                if action == "turn":
                    time.sleep(0.1)
                    power = random.uniform(90, 100)
                    cmd_q.put({"action": "dash", "power": power, "direction": base_direction})
                    try:
                        response = resp_q.get(timeout=0.5)
                    except multiprocessing.queues.Empty:
                        pass
                
            except Exception as e:
                print(f"Error in player behavior for {name}: {e}")
            
            # Shorter sleep time to increase activity
            time.sleep(random.uniform(0.2, 0.5))
    
    # Function for goalie behavior
    def goalie_behavior(goalie_agent, is_left_team):
        """Execute more active goalie behavior."""
        _, cmd_q, resp_q, name = goalie_agent
        
        # Goalie position coordinates
        x = goal_x_left + 2 if is_left_team else goal_x_right - 2
        base_direction = 0 if is_left_team else 180
        
        while True:
            try:
                # More active goalie behavior
                action = random.choices(["dash", "turn", "catch", "move_sideways"], 
                                       weights=[0.5, 0.2, 0.1, 0.2])[0]
                
                if action == "dash":
                    # More powerful dashes for goalies
                    power = random.uniform(60, 100)
                    direction_variation = random.uniform(-30, 30)
                    direction = (base_direction + direction_variation) % 360
                    cmd_q.put({"action": "dash", "power": power, "direction": direction})
                
                elif action == "turn":
                    moment = random.uniform(-60, 60)
                    cmd_q.put({"action": "turn", "moment": moment})
                
                elif action == "catch":
                    direction = random.uniform(-45, 45)
                    cmd_q.put({"action": "catch", "direction": direction})
                
                elif action == "move_sideways":
                    y_offset = random.uniform(-10, 10)
                    cmd_q.put({"action": "move", "position": (x, y_offset)})
                    # Follow up with a dash to activate
                    time.sleep(0.2)
                    cmd_q.put({"action": "dash", "power": 80, "direction": base_direction})
                
                # Process response
                try:
                    response = resp_q.get(timeout=0.5)
                except multiprocessing.queues.Empty:
                    pass
                
            except Exception as e:
                print(f"Error in goalie behavior for {name}: {e}")
            
            time.sleep(random.uniform(0.3, 0.7))
    
    # Start player threads
    print("Starting gameplay with enhanced movement...")
    threads = []
    
    # Start field player threads
    for i, agent in enumerate(left_agents[1:], 0):
        role = i % 3  # 0: defender, 1: midfielder, 2: forward
        thread = Thread(target=active_player_behavior, args=(agent, True, role))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    for i, agent in enumerate(right_agents[1:], 0):
        role = i % 3
        thread = Thread(target=active_player_behavior, args=(agent, False, role))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Start goalie threads
    left_goalie_thread = Thread(target=goalie_behavior, args=(left_agents[0], True))
    left_goalie_thread.daemon = True
    left_goalie_thread.start()
    threads.append(left_goalie_thread)
    
    right_goalie_thread = Thread(target=goalie_behavior, args=(right_agents[0], False))
    right_goalie_thread.daemon = True
    right_goalie_thread.start()
    threads.append(right_goalie_thread)
    
    # Periodic forceful actions to ensure game activity
    def ensure_activity():
        """Periodically force activities to keep the game moving."""
        while True:
            try:
                # Select random players to take forceful action
                if left_agents[1:]:
                    _, cmd_q, resp_q, name = random.choice(left_agents[1:])
                    cmd_q.put({"action": "dash", "power": 100, "direction": 0})
                    try:
                        resp_q.get(timeout=0.5)
                    except multiprocessing.queues.Empty:
                        pass
                
                if right_agents[1:]:
                    _, cmd_q, resp_q, name = random.choice(right_agents[1:])
                    cmd_q.put({"action": "dash", "power": 100, "direction": 180})
                    try:
                        resp_q.get(timeout=0.5)
                    except multiprocessing.queues.Empty:
                        pass
                
                # Force a goalie to move forward occasionally
                team = random.choice([left_agents, right_agents])
                if team:
                    _, cmd_q, resp_q, name = team[0]  # Goalie
                    direction = 0 if team == left_agents else 180
                    cmd_q.put({"action": "dash", "power": 70, "direction": direction})
                    try:
                        resp_q.get(timeout=0.5)
                    except multiprocessing.queues.Empty:
                        pass
                
            except Exception as e:
                print(f"Error in activity enforcement: {e}")
            
            time.sleep(5)  # Force activity every 5 seconds
    
    # Start activity enforcement thread
    activity_thread = Thread(target=ensure_activity)
    activity_thread.daemon = True
    activity_thread.start()
    threads.append(activity_thread)
    
    # Game timer with periodic reporting
    print(f"Game started! Will run for {game_duration} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < game_duration:
            remaining = int(game_duration - (time.time() - start_time))
            if remaining % 30 == 0 and remaining > 0:  # Report every 30 seconds
                print(f"Game time remaining: {remaining} seconds")
                
                # Force a central player to dash toward ball
                if len(left_agents) > 2:  # Use a midfielder
                    _, cmd_q, resp_q, name = left_agents[2]  # Midfielder
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
        print("Shutting down all agents...")
        for proc, cmd_q, _, name in agents:
            cmd_q.put({"action": "exit"})
            proc.join(timeout=1)
            if proc.is_alive():
                proc.terminate()
        
        print("Game ended.")

# Update the main section to use the new function
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RoboCup Soccer Simulator Client')
    parser.add_argument('--players_count', type=int, default=3,
                        help='Number of field players in each team (excluding goalie) (default: 3)')
    parser.add_argument('--game_duration', type=int, default=300,
                        help='Duration of the game in seconds (default: 300)')
    args = parser.parse_args()
    
    # Run the complete game simulation
    print(f"Starting RoboCup simulation with {args.players_count} field players per team plus goalies...")
    print(f"Game will run for {args.game_duration} seconds.")
    conduct_game(players_count=args.players_count, game_duration=args.game_duration)