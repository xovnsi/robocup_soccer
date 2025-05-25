import torch
import torch.nn.functional as F
import random
from constants import *
import pygame

class ImprovedFootballSimulation:
    def __init__(self, num_players_per_team=3, use_rendering=False):  # Reduced to 3 players for easier learning
        # Initialize device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize simulation time
        self.time_factor = 1.0
        self.simulation_time = 0.0

        # Initialize field
        self.field_width = torch.tensor(FIELD_WIDTH, device=self.device)
        self.field_height = torch.tensor(FIELD_HEIGHT, device=self.device)

        # Initialize ball
        self.ball_pos = torch.tensor([FIELD_WIDTH / 2, FIELD_HEIGHT / 2], dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.prev_ball_pos = self.ball_pos.clone()

        # Initialize teams
        self.num_players_per_team = num_players_per_team
        self.team_a_pos = self._initialize_team_positions(team="A")
        self.team_b_pos = self._initialize_team_positions(team="B")
        self.team_a_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)
        self.team_b_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)

        # Store previous positions for reward calculation
        self.prev_team_a_pos = self.team_a_pos.clone()
        self.prev_team_b_pos = self.team_b_pos.clone()

        # Initialize ball possession info
        self.ball_possession = {"team": None, "player_id": None}
        self.last_possession = {"team": None, "player_id": None}

        # Scores
        self.score_a = 0
        self.score_b = 0

        # Use pygame visualization?
        self.use_rendering = use_rendering

        # Initialize pygame for visualization if needed
        if self.use_rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Football Simulation 2D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 12)

        # Tracking rewards and game state
        self.reset_reward_tracking()

        # Episode tracking
        self.current_episode = 0
        self.max_episode_steps = 1000  # Increased episode length

    def reset_reward_tracking(self):
        """Reset all reward-related tracking variables"""
        self.last_team_with_ball = None
        self.successful_pass = False
        self.last_ball_pos = self.ball_pos.clone()
        self.steps_since_last_touch = 0
        self.current_step = 0
        self.last_ball_touched_by = {"team": None, "player_id": None}

    def reset(self):
        """Reset the simulation for a new episode"""
        # Reset ball to random position near center
        offset_x = random.uniform(-10, 10)
        offset_y = random.uniform(-10, 10)
        self.ball_pos = torch.tensor([FIELD_WIDTH / 2 + offset_x, FIELD_HEIGHT / 2 + offset_y],
                                     dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.prev_ball_pos = self.ball_pos.clone()

        # Reset teams with some randomization
        self.team_a_pos = self._initialize_team_positions(team="A")
        self.team_b_pos = self._initialize_team_positions(team="B")
        self.team_a_vel = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)
        self.team_b_vel = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)

        self.prev_team_a_pos = self.team_a_pos.clone()
        self.prev_team_b_pos = self.team_b_pos.clone()

        # Reset possession
        self.ball_possession = {"team": None, "player_id": None}
        self.last_possession = {"team": None, "player_id": None}

        # Don't reset scores - keep cumulative

        # Reset simulation time
        self.simulation_time = 0.0

        # Reset reward tracking
        self.reset_reward_tracking()
        self.current_episode += 1
        self.current_step = 0

        # Return initial state
        return self.get_state_tensor()

    def _initialize_team_positions(self, team):
        """Initialize player positions with better spacing"""
        positions = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)

        if team == "A":  # Left team
            # Goalkeeper
            positions[0] = torch.tensor([8.0, FIELD_HEIGHT / 2 + random.uniform(-5, 5)])
            # Field players
            for i in range(1, self.num_players_per_team):
                x = random.uniform(15, FIELD_WIDTH / 2 - 10)
                y = random.uniform(15, FIELD_HEIGHT - 15)
                positions[i] = torch.tensor([x, y])
        else:  # Right team
            # Goalkeeper
            positions[0] = torch.tensor([FIELD_WIDTH - 8.0, FIELD_HEIGHT / 2 + random.uniform(-5, 5)])
            # Field players
            for i in range(1, self.num_players_per_team):
                x = random.uniform(FIELD_WIDTH / 2 + 10, FIELD_WIDTH - 15)
                y = random.uniform(15, FIELD_HEIGHT - 15)
                positions[i] = torch.tensor([x, y])

        return positions

    def set_team_actions(self, team, actions_tensor):
        """Process and set team actions"""
        actions = actions_tensor.view(self.num_players_per_team, 6)

        if team == "A":
            self.team_a_actions = actions
        else:
            self.team_b_actions = actions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _apply_movement(self, positions, velocities, actions):
        """Apply movement actions to team positions and velocities"""
        # Extract movement actions (first 2 components of each player's action)
        movement_actions = actions[:, :2]  # Shape: (num_players, 2)

        # Update velocities based on movement actions
        velocities = velocities + movement_actions * 0.1  # Small acceleration factor

        # Calculate speeds (magnitude of velocity vectors)
        speeds = torch.norm(velocities, dim=1, keepdim=True)  # Shape: (num_players, 1)

        # Create mask for players exceeding max speed
        speed_mask = speeds > MAX_SPEED  # Shape: (num_players, 1)

        # Clamp velocities that exceed max speed - using torch.where for safety
        normalized_velocities = velocities / speeds * MAX_SPEED
        velocities = torch.where(speed_mask, normalized_velocities, velocities)

        # Update positions
        positions = positions + velocities * 0.1  # Time step factor

        # Keep players within field boundaries
        positions[:, 0] = torch.clamp(positions[:, 0], -FIELD_WIDTH / 2, FIELD_WIDTH / 2)
        positions[:, 1] = torch.clamp(positions[:, 1], -FIELD_HEIGHT / 2, FIELD_HEIGHT / 2)

        return positions, velocities

    def _update_ball(self):
        """Update ball with improved physics and goal detection"""
        self.prev_ball_pos = self.ball_pos.clone()
        goal_scored = False
        goal_team = None

        # Check ball interaction with players
        for team, positions, actions in [
            ("A", self.team_a_pos, self.team_a_actions),
            ("B", self.team_b_pos, self.team_b_actions)
        ]:
            for player_id in range(self.num_players_per_team):
                dist_to_ball = torch.norm(positions[player_id] - self.ball_pos)

                if dist_to_ball < CONTROL_DISTANCE:
                    self.ball_possession = {"team": team, "player_id": player_id}
                    self.last_ball_touched_by = {"team": team, "player_id": player_id}

                    # Check for kick action (3rd component > 0.5)
                    if actions[player_id, 2] > 0.5:
                        kick_power = torch.sigmoid(actions[player_id, 3]) * MAX_KICK_POWER
                        kick_direction = actions[player_id, 4:6]

                        # Normalize kick direction
                        kick_norm = torch.norm(kick_direction)
                        if kick_norm > 0.1:
                            kick_direction = kick_direction / kick_norm
                        else:
                            # Default kick direction toward opponent goal
                            if team == "A":
                                kick_direction = torch.tensor([1.0, 0.0], device=self.device)
                            else:
                                kick_direction = torch.tensor([-1.0, 0.0], device=self.device)

                        # Apply kick
                        self.ball_vel = kick_direction * kick_power
                        break

        # Update ball position
        self.ball_pos = self.ball_pos + self.ball_vel * TIME_STEP
        self.ball_vel = self.ball_vel * BALL_DECELERATION

        # Goal detection
        goal_width = 7.32
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width

        if self.ball_pos[0] <= 0:
            if goal_y_start <= self.ball_pos[1] <= goal_y_end:
                goal_scored = True
                goal_team = "B"
                self.score_b += 1
            # Reset ball position
            self.ball_pos = torch.tensor([FIELD_WIDTH / 2, FIELD_HEIGHT / 2], device=self.device)
            self.ball_vel = torch.zeros(2, device=self.device)

        elif self.ball_pos[0] >= FIELD_WIDTH:
            if goal_y_start <= self.ball_pos[1] <= goal_y_end:
                goal_scored = True
                goal_team = "A"
                self.score_a += 1
            # Reset ball position
            self.ball_pos = torch.tensor([FIELD_WIDTH / 2, FIELD_HEIGHT / 2], device=self.device)
            self.ball_vel = torch.zeros(2, device=self.device)

        # Boundary bouncing for top/bottom
        if self.ball_pos[1] <= 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] = -self.ball_vel[1] * 0.7
        elif self.ball_pos[1] >= FIELD_HEIGHT:
            self.ball_pos[1] = FIELD_HEIGHT
            self.ball_vel[1] = -self.ball_vel[1] * 0.7

        return goal_scored, goal_team

    def calculate_improved_rewards(self, goal_scored, goal_team):
        """Calculate more comprehensive rewards"""
        reward_a = 0
        reward_b = 0

        # Goal rewards (primary objective)
        if goal_scored:
            if goal_team == "A":
                reward_a += GOAL_REWARD
                reward_b += OPPONENT_GOAL_PENALTY
            else:
                reward_a += OPPONENT_GOAL_PENALTY
                reward_b += GOAL_REWARD

        # Ball possession rewards
        if self.ball_possession["team"] == "A":
            reward_a += BALL_POSSESSION_REWARD
        elif self.ball_possession["team"] == "B":
            reward_b += BALL_POSSESSION_REWARD

        # Ball progress toward goal (shaped reward)
        ball_progress_x = self.ball_pos[0] - self.prev_ball_pos[0]
        if abs(ball_progress_x) > 0.1:  # Only reward significant movement
            reward_a += ball_progress_x * BALL_PROGRESS_REWARD / FIELD_WIDTH
            reward_b -= ball_progress_x * BALL_PROGRESS_REWARD / FIELD_WIDTH

        # Reward for players moving toward the ball (encourage engagement)
        for i in range(self.num_players_per_team):
            # Team A
            prev_dist_a = torch.norm(self.prev_team_a_pos[i] - self.prev_ball_pos)
            curr_dist_a = torch.norm(self.team_a_pos[i] - self.ball_pos)
            if curr_dist_a < prev_dist_a:
                reward_a += MOVE_TO_BALL_REWARD

            # Team B
            prev_dist_b = torch.norm(self.prev_team_b_pos[i] - self.prev_ball_pos)
            curr_dist_b = torch.norm(self.team_b_pos[i] - self.ball_pos)
            if curr_dist_b < prev_dist_b:
                reward_b += MOVE_TO_BALL_REWARD

        # Successful pass detection and reward
        if (self.last_possession["team"] is not None and
                self.ball_possession["team"] == self.last_possession["team"] and
                self.ball_possession["player_id"] != self.last_possession["player_id"]):
            if self.ball_possession["team"] == "A":
                reward_a += SUCCESSFUL_PASS_REWARD
            else:
                reward_b += SUCCESSFUL_PASS_REWARD

        return reward_a, reward_b

    def step(self, action_a=None, action_b=None):
        """Improved step function"""
        # Store previous positions
        self.prev_team_a_pos = self.team_a_pos.clone()
        self.prev_team_b_pos = self.team_b_pos.clone()
        self.last_possession = self.ball_possession.copy()

        # Apply actions
        if action_a is not None:
            self.set_team_actions("A", action_a)
        if action_b is not None:
            self.set_team_actions("B", action_b)

        # Update player positions
        self.team_a_pos, self.team_a_vel = self._apply_movement(
            self.team_a_pos, self.team_a_vel, self.team_a_actions)
        self.team_b_pos, self.team_b_vel = self._apply_movement(
            self.team_b_pos, self.team_b_vel, self.team_b_actions)

        # Update ball
        goal_scored, goal_team = self._update_ball()

        # Calculate rewards
        reward_a, reward_b = self.calculate_improved_rewards(goal_scored, goal_team)

        # Update time and step counter
        self.simulation_time += TIME_STEP
        self.current_step += 1

        # Get next state
        next_state = self.get_state_tensor()

        # Episode termination conditions
        done = (self.current_step >= self.max_episode_steps or
                goal_scored)  # End episode on goal for faster learning

        return next_state, (reward_a, reward_b), done, {
            "goal_scored": goal_scored,
            "goal_team": goal_team,
            "ball_possession": self.ball_possession,
            "score": (self.score_a, self.score_b)
        }

    def get_state_tensor(self):
        """Get normalized state representation"""
        state_dim = 4 + 4 * self.num_players_per_team * 2
        state = torch.zeros(1, state_dim, device=self.device)

        # Ball state [x, y, vx, vy] - normalized
        state[0, 0] = self.ball_pos[0] / FIELD_WIDTH
        state[0, 1] = self.ball_pos[1] / FIELD_HEIGHT
        state[0, 2] = self.ball_vel[0] / MAX_KICK_POWER
        state[0, 3] = self.ball_vel[1] / MAX_KICK_POWER

        # Team A players
        for i in range(self.num_players_per_team):
            idx = 4 + i * 4
            state[0, idx] = self.team_a_pos[i, 0] / FIELD_WIDTH
            state[0, idx + 1] = self.team_a_pos[i, 1] / FIELD_HEIGHT
            state[0, idx + 2] = self.team_a_vel[i, 0] / MAX_SPEED
            state[0, idx + 3] = self.team_a_vel[i, 1] / MAX_SPEED

        # Team B players
        for i in range(self.num_players_per_team):
            idx = 4 + self.num_players_per_team * 4 + i * 4
            state[0, idx] = self.team_b_pos[i, 0] / FIELD_WIDTH
            state[0, idx + 1] = self.team_b_pos[i, 1] / FIELD_HEIGHT
            state[0, idx + 2] = self.team_b_vel[i, 0] / MAX_SPEED
            state[0, idx + 3] = self.team_b_vel[i, 1] / MAX_SPEED

        return state

    def render(self):
        """Render the simulation"""
        if not self.use_rendering:
            return

        scale_x = SCREEN_WIDTH / FIELD_WIDTH
        scale_y = SCREEN_HEIGHT / FIELD_HEIGHT

        self.screen.fill(GREEN)

        # Field markings
        pygame.draw.rect(self.screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH / 2, 0), (SCREEN_WIDTH / 2, SCREEN_HEIGHT), 2)
        pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH / 2), int(SCREEN_HEIGHT / 2)), 50, 2)

        # Goals
        goal_width = 7.32
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width

        pygame.draw.line(self.screen, WHITE, (0, goal_y_start * scale_y), (0, goal_y_end * scale_y), 5)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH, goal_y_start * scale_y),
                         (SCREEN_WIDTH, goal_y_end * scale_y), 5)

        # Players
        for i in range(self.num_players_per_team):
            # Team A (red)
            pos_a = self.team_a_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, RED, (int(pos_a[0] * scale_x), int(pos_a[1] * scale_y)), 8)

            # Team B (blue)
            pos_b = self.team_b_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, BLUE, (int(pos_b[0] * scale_x), int(pos_b[1] * scale_y)), 8)

        # Ball
        ball_pos = self.ball_pos.cpu().numpy()
        pygame.draw.circle(self.screen, YELLOW, (int(ball_pos[0] * scale_x), int(ball_pos[1] * scale_y)), 6)

        # Info
        info_text = f"Score: A {self.score_a} - {self.score_b} B | Episode: {self.current_episode} | Step: {self.current_step}"
        info_surface = self.font.render(info_text, True, WHITE, BLACK)
        self.screen.blit(info_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)