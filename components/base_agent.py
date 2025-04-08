import socket
import time
from enum import Enum
import re


class TeamSide(Enum):
    LEFT = "SimpleTeam"
    RIGHT = "EnemyTeam"


class Role(Enum):
    DEFAULT = "default"
    STRIKER = "striker"
    GOALIE = "goalie"


class Agent:
    def __init__(self, team: TeamSide, role: Role = Role.DEFAULT):
        self.team = team
        self.role = role
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.server_addr = ('127.0.0.1', 6000)
        self.game_started = False  

    def init_connection(self):
        init_msg = f"(init {self.team.value} (version 15))"
        self.sock.sendto(init_msg.encode(), self.server_addr)
        data, _ = self.sock.recvfrom(4096)
        print(f"[{self.team.value} - {self.role.value}] Connected: {data.decode().strip()}")

    def parse_see(self, msg):
        """ Parse the see message to find the ball and its angle. """
        match = re.search(r"\(see \d+ ((?:\([^\)]+\) ?)+)\)", msg)
        if not match:
            return None, None

        objects = match.group(1)

        # Find ball info
        ball_match = re.search(r"\(ball (-?\d+\.\d+) (-?\d+\.\d+)", objects)
        if ball_match:
            ball_angle = float(ball_match.group(1))  # Angle to ball
            ball_distance = float(ball_match.group(2))  # Distance to ball
            return ball_angle, ball_distance  # Return both angle and distance

        return None, None  # No ball detected

    def move_agents(self):
        """ This method could be expanded for general movement logic. """
        if self.role == Role.STRIKER:
            self.sock.sendto("(dash 50)".encode(), self.server_addr)

    def striker_behavior(self):
        """ Main behavior for the striker agent. """
        try:
            while True:
                msg, _ = self.sock.recvfrom(4096)
                decoded = msg.decode()

                # Print all incoming messages for debugging
                print(f"[Agent Message] {decoded}")  # Print every message received

                if decoded.startswith("(see"):
                    ball_angle, ball_distance = self.parse_see(decoded)

                    if ball_angle is not None:
                        print(f"[Striker] Ball at angle: {ball_angle:.2f}, distance: {ball_distance:.2f}")

                        if ball_distance < 0.5:  # Close enough to kick
                            self.sock.sendto("(kick 100 0)".encode(), self.server_addr)
                            print("[Striker] Kicked the ball!")
                        elif abs(ball_angle) > 10:
                            # Turn toward the ball if it's at a significant angle
                            self.sock.sendto(f"(turn {ball_angle})".encode(), self.server_addr)
                        else:
                            # Dash toward the ball if the angle is small enough
                            self.sock.sendto("(dash 80)".encode(), self.server_addr)

                elif decoded.startswith("(sense_body"):
                    if "before_kick_off" in decoded:
                        print("[Striker] Waiting for kickoff...")
                        continue  # Game has not started
                    else:
                        self.game_started = True
                        print("[Striker] Game started! Moving agents.")
                        self.move_agents()  # Move agents once the game starts

                elif decoded.startswith("(hear"):
                    continue  # Handle hear commands if necessary

        except KeyboardInterrupt:
            print("Striker shutting down.")
        finally:
            self.sock.close()



    def run(self):
        self.init_connection()

        while True:
            if self.role == Role.STRIKER:
                self.striker_behavior()
            else:
                # Default agents just move forward
                self.sock.sendto("(dash 50)".encode(), self.server_addr)
                time.sleep(0.1)
