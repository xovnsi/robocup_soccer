FIELD_WIDTH = 105.0  # width in meters
FIELD_HEIGHT = 68.0  # height in meters
MAX_SPEED = 8.0  # max player speed (m/s)
MAX_KICK_POWER = 25.0  # max kick power (m/s)
BALL_DECELERATION = 0.95  # ball slowdown factor
PLAYER_ACCELERATION = 2.0  # player acceleration
PLAYER_DECELERATION = 0.9  # player slowdown factor
CONTROL_DISTANCE = 1.5  # ball control distance
TIME_STEP = 0.05  # simulation time step (seconds)
SCREEN_WIDTH = 800  # screen width in pixels
SCREEN_HEIGHT = 600  # screen height in pixels

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# IMPROVED Reinforcement Learning parameters
GAMMA = 0.99  # discount factor
MEMORY_SIZE = 50000  # increased replay memory size
BATCH_SIZE = 128  # increased batch size
LR_ACTOR = 0.0001  # reduced learning rate for actor
LR_CRITIC = 0.001  # learning rate for critic
NOISE_SCALE_START = 0.3  # initial noise scale
NOISE_SCALE_END = 0.05  # final noise scale
NOISE_DECAY = 0.9995  # noise decay factor
TAU = 0.005  # increased for soft update

# IMPROVED Reward weights - more balanced
GOAL_REWARD = 100.0
OPPONENT_GOAL_PENALTY = -100.0
BALL_POSSESSION_REWARD = 0.1
BALL_TOUCH_REWARD = 1.0
MOVE_TO_BALL_REWARD = 0.05
OPPONENT_HALF_REWARD = 0.02
SUCCESSFUL_PASS_REWARD = 2.0
BALL_PROGRESS_REWARD = 5.0
