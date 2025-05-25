import torch.nn as nn
import torch.nn.functional as F
import torch

class ImprovedTeamAgent(nn.Module):
    """Improved Neural network that controls all players of a team"""

    def __init__(self, num_players, state_dim, action_dim_per_player):
        super(ImprovedTeamAgent, self).__init__()

        self.num_players = num_players
        self.state_dim = state_dim
        self.action_dim_per_player = action_dim_per_player
        self.total_action_dim = num_players * action_dim_per_player

        # Shared feature extraction layers with batch normalization
        self.fc1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        # Action output layers for each player
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim_per_player),
                nn.Tanh()  # Output in [-1, 1] range
            ) for _ in range(num_players)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        # Get actions for each player
        player_actions = []
        for i in range(self.num_players):
            actions = self.action_heads[i](x)
            player_actions.append(actions)

        # Combine all player actions
        team_actions = torch.cat(player_actions, dim=1)
        return team_actions

