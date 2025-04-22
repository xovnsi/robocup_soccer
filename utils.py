import multiprocessing
import time

def create_rcss_agent(team, player_num, is_goalie=False):
    """Tworzy i uruchamia agenta RCSS."""
    from client import RCSSClient
    
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

def calculate_formations(players_count):
    """Calculate formation positions for both teams."""
    # Field dimensions and positions
    field_length = 105
    field_width = 68
    goal_x_left = -52
    goal_x_right = 52
    
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
    
    return left_positions, right_positions