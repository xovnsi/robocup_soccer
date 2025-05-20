import torch
from train import FootballSimulation, TeamAgent

def main():
    sim = FootballSimulation(num_players_per_team=5, use_rendering=True)

    policy_net_a = TeamAgent(5, 4 + 4*5*2, 6).to(sim.device)
    policy_net_a.load_state_dict(torch.load('policy_net_a.pth', map_location=sim.device))
    policy_net_a.eval()

    policy_net_b = TeamAgent(5, 4 + 4*5*2, 6).to(sim.device)
    policy_net_b.load_state_dict(torch.load('policy_net_b.pth', map_location=sim.device))
    policy_net_b.eval()

    def normalize_actions(actions):
        move_dir = actions[:, :2]
        norms = torch.norm(move_dir, dim=1, keepdim=True)
        move_dir = move_dir / (norms + 1e-8)
        actions[:, :2] = move_dir
        return actions

    running = True
    while running:
        running = sim.check_for_events()

        state = sim.get_state_tensor()
        with torch.no_grad():
            action_a = policy_net_a(state.to(sim.device))
            action_b = policy_net_b(state.to(sim.device))

        action_a = action_a.view(sim.num_players_per_team, 6).cpu()
        action_b = action_b.view(sim.num_players_per_team, 6).cpu()

        action_a = normalize_actions(action_a)
        action_b = normalize_actions(action_b)

        sim.set_team_actions('A', action_a)
        sim.set_team_actions('B', action_b)

        sim.update()
        sim.render()

    sim.close()  # or pygame.quit() if close() is not defined

if __name__ == "__main__":
    main()
