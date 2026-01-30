import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.sequential(x)
    
if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    model = DQN(input_dim=state_dim, hidden_dim=128, output_dim=action_dim)
    state = torch.randn(10, state_dim)
    output = model(state)
    print(output)