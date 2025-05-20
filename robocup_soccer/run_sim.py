import torch
import pygame
from train import FootballSimulation, TeamAgent

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    sim = FootballSimulation(num_players_per_team=5)

    # Load trained model
    policy_net_a = TeamAgent(5, 4 + 4*5*2, 6).to(sim.device)
    policy_net_a.load_state_dict(torch.load('policy_net_a.pth', map_location=sim.device))
    policy_net_a.eval()

    # Load model for team B
    policy_net_b = TeamAgent(5, 4 + 4*5*2, 6).to(sim.device)
    policy_net_b.load_state_dict(torch.load('policy_net_b.pth', map_location=sim.device))
    policy_net_b.eval()


    running = True
    while running:
        running = sim.check_for_events()

        state = sim.get_state_tensor()
        with torch.no_grad():
            action_a = policy_net_a(state.to(sim.device))
            action_b = policy_net_b(state.to(sim.device))
        
        sim.set_team_actions('A', action_a)
        
        sim.set_team_actions('B', action_b)
        
        sim.update()
        sim.render()

    pygame.quit()

if __name__ == "__main__":
    main()
