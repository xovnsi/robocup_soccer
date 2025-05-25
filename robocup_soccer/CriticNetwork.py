from torch import nn
import torch
import torch.nn.functional as F


class ImprovedCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ImprovedCriticNetwork, self).__init__()

        # State processing
        self.state_fc1 = nn.Linear(state_dim, 256)
        self.state_fc2 = nn.Linear(256, 128)

        # Action processing
        self.action_fc1 = nn.Linear(action_dim, 128)

        # Combined processing
        self.combined_fc1 = nn.Linear(256, 256)  # 128 + 128
        self.combined_fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        # Process state
        s = F.relu(self.bn1(self.state_fc1(state)))
        s = F.relu(self.bn2(self.state_fc2(s)))

        # Process action
        a = F.relu(self.action_fc1(action))

        # Combine state and action
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.bn3(self.combined_fc1(x)))
        x = F.relu(self.bn4(self.combined_fc2(x)))
        q_value = self.output(x)
        return q_value