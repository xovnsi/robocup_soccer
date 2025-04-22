import random
import time
import multiprocessing
from threading import Thread

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

def goalie_behavior(goalie_agent, is_left_team):
    """Execute more active goalie behavior."""
    _, cmd_q, resp_q, name = goalie_agent
    
    # Goalie position coordinates
    goal_x_left = -52
    goal_x_right = 52
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

def ensure_activity(left_agents, right_agents):
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