from multiprocessing import Process
from components.base_agent import Agent, TeamSide, Role
import argparse

def launch_agent(team, role=Role.DEFAULT):
    agent = Agent(team, role)
    agent.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=3)
    args = parser.parse_args()

    agents = []

    for i in range(args.players):
        role = Role.STRIKER if i == 0 else Role.DEFAULT
        agents.append(Process(target=launch_agent, args=(TeamSide.LEFT, role)))

        # You can add striker for enemy team too if you want:
        role = Role.STRIKER if i == 0 else Role.DEFAULT
        agents.append(Process(target=launch_agent, args=(TeamSide.RIGHT, role)))

    for p in agents:
        p.start()
    for p in agents:
        p.join()

if __name__ == "__main__":
    main()
