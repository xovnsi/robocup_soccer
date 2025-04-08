from components.base_agent import TeamSide


class Coach:
    def __init__(self, team: TeamSide):
        self.team = team

    def send_command(self, command):
        pass

    def game_started(self):
        pass

