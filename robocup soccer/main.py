import torch
import pygame
import random
import math
import numpy as np
import time

# Stałe symulacji
FIELD_WIDTH = 105.0  # szerokość boiska w metrach
FIELD_HEIGHT = 68.0  # wysokość boiska w metrach
MAX_SPEED = 8.0      # maksymalna prędkość zawodnika (m/s)
MAX_KICK_POWER = 25.0  # maksymalna siła kopnięcia (m/s)
BALL_DECELERATION = 0.95  # współczynnik spowolnienia piłki
PLAYER_ACCELERATION = 2.0  # przyspieszenie zawodnika
PLAYER_DECELERATION = 0.9  # współczynnik spowolnienia zawodnika
CONTROL_DISTANCE = 1.5     # dystans kontroli piłki przez zawodnika
TIME_STEP = 0.05           # krok czasowy symulacji (w sekundach)
SCREEN_WIDTH = 800         # szerokość ekranu w pikselach
SCREEN_HEIGHT = 600        # wysokość ekranu w pikselach

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class FootballSimulation:
    def __init__(self, num_players_per_team=5):
        # Inicjalizacja urządzenia (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Używanie urządzenia: {self.device}")
        
        # Inicjalizacja czasu symulacji
        self.time_factor = 1.0  # 1.0 = czas normalny, > 1.0 = przyspieszony, < 1.0 = spowolniony
        self.simulation_time = 0.0
        
        # Inicjalizacja boiska
        self.field_width = torch.tensor(FIELD_WIDTH, device=self.device)
        self.field_height = torch.tensor(FIELD_HEIGHT, device=self.device)
        
        # Inicjalizacja piłki
        self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        
        # Inicjalizacja drużyn
        self.num_players_per_team = num_players_per_team
        self.team_a_pos = self._initialize_team_positions(team="A")
        self.team_b_pos = self._initialize_team_positions(team="B")
        self.team_a_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)
        self.team_b_vel = torch.zeros((num_players_per_team, 2), dtype=torch.float32, device=self.device)
        
        # Inicjalizacja wektora akcji dla każdego gracza
        # [kierunek_x, kierunek_y, czy_kopać, siła_kopnięcia, kierunek_kopnięcia_x, kierunek_kopnięcia_y]
        self.team_a_actions = torch.zeros((num_players_per_team, 6), dtype=torch.float32, device=self.device)
        self.team_b_actions = torch.zeros((num_players_per_team, 6), dtype=torch.float32, device=self.device)
        
        # Inicjalizacja informacji o posiadaniu piłki
        self.ball_possesion = {"team": None, "player_id": None}
        
        # Licznik wyników
        self.score_a = 0
        self.score_b = 0
        
        # Czy używać wizualizacji Pygame
        self.use_rendering = True
        
        # Inicjalizacja pygame do wizualizacji
        if self.use_rendering:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Symulacja Piłki Nożnej 2D")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 12)

    def _initialize_team_positions(self, team):
        """Inicjalizacja pozycji startowych zawodników drużyny"""
        positions = torch.zeros((self.num_players_per_team, 2), dtype=torch.float32, device=self.device)
        
        if team == "A":  # Drużyna po lewej stronie
            x_base = FIELD_WIDTH / 4
            positions[0] = torch.tensor([5.0, FIELD_HEIGHT/2])  # Bramkarz
        else:  # Drużyna po prawej stronie
            x_base = 3 * FIELD_WIDTH / 4
            positions[0] = torch.tensor([FIELD_WIDTH - 5.0, FIELD_HEIGHT/2])  # Bramkarz
        
        # Rozstawienie pozostałych zawodników
        for i in range(1, self.num_players_per_team):
            if team == "A":
                positions[i] = torch.tensor([
                    x_base + random.uniform(-10, 10),
                    random.uniform(10, FIELD_HEIGHT - 10)
                ])
            else:
                positions[i] = torch.tensor([
                    x_base + random.uniform(-10, 10),
                    random.uniform(10, FIELD_HEIGHT - 10)
                ])
        
        return positions
    
    def set_player_action(self, team, player_id, direction_x, direction_y, 
                         kick=False, kick_power=0.0, kick_direction_x=0.0, kick_direction_y=0.0):
        """Ustawia akcję dla danego zawodnika"""
        action = torch.tensor([direction_x, direction_y, float(kick), 
                              kick_power, kick_direction_x, kick_direction_y], 
                              dtype=torch.float32, device=self.device)
        
        if team == "A":
            self.team_a_actions[player_id] = action
        else:
            self.team_b_actions[player_id] = action
    
    def _apply_movement(self, positions, velocities, actions):
        """Zastosuj ruchy zawodników na podstawie ich akcji"""
        # Normalizuj kierunek ruchu
        directions = actions[:, :2]
        norms = torch.norm(directions, dim=1, keepdim=True)
        # Utwórz maskę i usuń wymiar keepdim, aby pasował do kształtu tensorów
        mask = (norms > 0).squeeze(-1)
        normalized_directions = torch.zeros_like(directions)
        
        # Bezpieczna operacja normalizacji tylko dla niezerowych kierunków
        for i in range(len(directions)):
            if mask[i]:
                normalized_directions[i] = directions[i] / norms[i]
        
        # Zaktualizuj prędkości
        accelerations = normalized_directions * PLAYER_ACCELERATION
        velocities = velocities * PLAYER_DECELERATION + accelerations * TIME_STEP * self.time_factor
        
        # Ogranicz maksymalną prędkość
        vel_norms = torch.norm(velocities, dim=1, keepdim=True)
        vel_mask = (vel_norms > MAX_SPEED).squeeze(-1)
        
        # Bezpieczna operacja ograniczania prędkości
        for i in range(len(velocities)):
            if vel_mask[i]:
                velocities[i] = velocities[i] / vel_norms[i] * MAX_SPEED
        
        # Zaktualizuj pozycje
        positions = positions + velocities * TIME_STEP * self.time_factor
        
        # Ogranicz pozycje do granic boiska
        positions[:, 0] = torch.clamp(positions[:, 0], 0, self.field_width)
        positions[:, 1] = torch.clamp(positions[:, 1], 0, self.field_height)
        
        return positions, velocities
    
    def _update_ball(self):
        """Aktualizacja pozycji i prędkości piłki"""
        # Sprawdź kopnięcia
        for team, positions, actions in [
            ("A", self.team_a_pos, self.team_a_actions),
            ("B", self.team_b_pos, self.team_b_actions)
        ]:
            for player_id in range(self.num_players_per_team):
                dist_to_ball = torch.norm(positions[player_id] - self.ball_pos)
                
                # Jeśli zawodnik jest wystarczająco blisko piłki
                if dist_to_ball < CONTROL_DISTANCE:
                    # Aktualizuj posiadanie piłki
                    self.ball_possesion = {"team": team, "player_id": player_id}
                    
                    # Jeśli zawodnik kopie piłkę
                    if actions[player_id, 2] > 0:
                        kick_power = torch.clamp(actions[player_id, 3], 0, MAX_KICK_POWER)
                        kick_direction = actions[player_id, 4:6]
                        if torch.norm(kick_direction) > 0:
                            kick_direction = kick_direction / torch.norm(kick_direction)
                        else:
                            # Domyślny kierunek, jeśli nie określono
                            if team == "A":
                                kick_direction = torch.tensor([1.0, 0.0], device=self.device)
                            else:
                                kick_direction = torch.tensor([-1.0, 0.0], device=self.device)
                        
                        # Zastosuj impuls do piłki
                        self.ball_vel = kick_direction * kick_power
                        break
                else:
                    # Jeśli nikt nie kontroluje piłki
                    if self.ball_possesion["team"] == team and self.ball_possesion["player_id"] == player_id:
                        self.ball_possesion = {"team": None, "player_id": None}
        
        # Aktualizuj pozycję piłki
        self.ball_pos = self.ball_pos + self.ball_vel * TIME_STEP * self.time_factor
        
        # Spowalniaj piłkę przez tarcie
        self.ball_vel = self.ball_vel * BALL_DECELERATION
        
        # Bramki i odbijanie piłki od granic
        goal_scored = False
        goal_team = None
        
        # Parametry bramki
        goal_width = 7.32  # szerokość bramki w metrach
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width
        
        # Sprawdź czy piłka wpadła do bramki
        if self.ball_pos[0] < 0:
            if goal_y_start < self.ball_pos[1] < goal_y_end:
                # Gol dla drużyny B
                goal_scored = True
                goal_team = "B"
                # Reset piłki na środek
                self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                          dtype=torch.float32, device=self.device)
                self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                # Odbij od ściany
                self.ball_pos[0] = 0
                self.ball_vel[0] = -self.ball_vel[0] * 0.7
        elif self.ball_pos[0] > self.field_width:
            if goal_y_start < self.ball_pos[1] < goal_y_end:
                # Gol dla drużyny A
                goal_scored = True
                goal_team = "A"
                # Reset piłki na środek
                self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                          dtype=torch.float32, device=self.device)
                self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
            else:
                # Odbij od ściany
                self.ball_pos[0] = self.field_width
                self.ball_vel[0] = -self.ball_vel[0] * 0.7
            
        if self.ball_pos[1] < 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] = -self.ball_vel[1] * 0.7
        elif self.ball_pos[1] > self.field_height:
            self.ball_pos[1] = self.field_height
            self.ball_vel[1] = -self.ball_vel[1] * 0.7
            
        return goal_scored, goal_team
    
    def update(self):
        """Jeden krok symulacji"""
        # Zaktualizuj pozycje i prędkości zawodników
        self.team_a_pos, self.team_a_vel = self._apply_movement(
            self.team_a_pos, self.team_a_vel, self.team_a_actions)
        self.team_b_pos, self.team_b_vel = self._apply_movement(
            self.team_b_pos, self.team_b_vel, self.team_b_actions)
        
        # Zaktualizuj piłkę i sprawdź gole
        goal_scored, goal_team = self._update_ball()
        
        # Aktualizuj wynik
        if goal_scored:
            if goal_team == "A":
                self.score_a += 1
            else:
                self.score_b += 1
        
        # Zaktualizuj czas symulacji
        self.simulation_time += TIME_STEP * self.time_factor
        
        return goal_scored, goal_team
    
    def get_state_tensor(self):
        """Zwraca stan gry jako jeden tensor"""
        # Stwórz tensor zawierający całą informację o stanie gry
        state = torch.zeros(1 + 2*self.num_players_per_team, 4, device=self.device)
        
        # Piłka [x, y, vx, vy]
        state[0, :2] = self.ball_pos
        state[0, 2:] = self.ball_vel
        
        # Drużyna A [x, y, vx, vy] dla każdego gracza
        state[1:self.num_players_per_team+1, :2] = self.team_a_pos
        state[1:self.num_players_per_team+1, 2:] = self.team_a_vel
        
        # Drużyna B [x, y, vx, vy] dla każdego gracza
        state[self.num_players_per_team+1:, :2] = self.team_b_pos
        state[self.num_players_per_team+1:, 2:] = self.team_b_vel
        
        return state
    
    def get_player_observation(self, team, player_id):
        """Zwraca tensor obserwacji z perspektywy danego zawodnika"""
        if team == "A":
            player_pos = self.team_a_pos[player_id]
        else:
            player_pos = self.team_b_pos[player_id]
        
        # Obserwacja zawiera:
        # - pozycję piłki względem zawodnika
        # - prędkość piłki
        # - pozycje wszystkich innych zawodników względem tego zawodnika
        
        observation = torch.zeros(1 + 2*self.num_players_per_team - 1, 4, device=self.device)
        
        # Piłka [x względne, y względne, vx, vy]
        observation[0, :2] = self.ball_pos - player_pos
        observation[0, 2:] = self.ball_vel
        
        # Pozostali zawodnicy z drużyny A
        obs_idx = 1
        for i in range(self.num_players_per_team):
            if team == "A" and i == player_id:
                continue  # pomijamy samego siebie
            observation[obs_idx, :2] = self.team_a_pos[i] - player_pos
            observation[obs_idx, 2:] = self.team_a_vel[i]
            obs_idx += 1
        
        # Zawodnicy z drużyny B
        for i in range(self.num_players_per_team):
            if team == "B" and i == player_id:
                continue  # pomijamy samego siebie
            observation[obs_idx, :2] = self.team_b_pos[i] - player_pos
            observation[obs_idx, 2:] = self.team_b_vel[i]
            obs_idx += 1
        
        return observation
    
    def render(self):
        """Renderuje symulację za pomocą pygame"""
        if not self.use_rendering:
            return
            
        # Oblicz współczynniki skalowania
        scale_x = SCREEN_WIDTH / FIELD_WIDTH
        scale_y = SCREEN_HEIGHT / FIELD_HEIGHT
        
        # Wypełnij tło (boisko)
        self.screen.fill(GREEN)
        
        # Narysuj linie boiska
        pygame.draw.rect(self.screen, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, WHITE, (SCREEN_WIDTH/2, 0), (SCREEN_WIDTH/2, SCREEN_HEIGHT), 2)
        pygame.draw.circle(self.screen, WHITE, (int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2)), 
                          int(min(FIELD_WIDTH, FIELD_HEIGHT)/10 * scale_x), 2)
        
        # Bramki
        goal_width = 7.32  # szerokość bramki w metrach
        goal_y_start = (FIELD_HEIGHT - goal_width) / 2
        goal_y_end = goal_y_start + goal_width
        
        # Lewa bramka
        pygame.draw.line(
            self.screen, 
            WHITE, 
            (0, goal_y_start * scale_y), 
            (0, goal_y_end * scale_y), 
            5
        )
        
        # Prawa bramka
        pygame.draw.line(
            self.screen, 
            WHITE, 
            (SCREEN_WIDTH, goal_y_start * scale_y), 
            (SCREEN_WIDTH, goal_y_end * scale_y), 
            5
        )
        
        # Narysuj zawodników drużyny A (czerwoni)
        for i in range(self.num_players_per_team):
            pos = self.team_a_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, RED, 
                              (int(pos[0] * scale_x), int(pos[1] * scale_y)), 
                              5)
            label = self.font.render(f"A{i}", True, WHITE)
            self.screen.blit(label, (int(pos[0] * scale_x) - 8, int(pos[1] * scale_y) - 8))
        
        # Narysuj zawodników drużyny B (niebiescy)
        for i in range(self.num_players_per_team):
            pos = self.team_b_pos[i].cpu().numpy()
            pygame.draw.circle(self.screen, BLUE, 
                              (int(pos[0] * scale_x), int(pos[1] * scale_y)), 
                              5)
            label = self.font.render(f"B{i}", True, WHITE)
            self.screen.blit(label, (int(pos[0] * scale_x) - 8, int(pos[1] * scale_y) - 8))
        
        # Narysuj piłkę
        ball_pos = self.ball_pos.cpu().numpy()
        pygame.draw.circle(self.screen, YELLOW, 
                          (int(ball_pos[0] * scale_x), int(ball_pos[1] * scale_y)), 
                          4)
        
        # Narysuj informacje o symulacji
        info_text = f"Czas: {self.simulation_time:.1f}s | Wynik: A {self.score_a} - {self.score_b} B | Mnożnik czasu: x{self.time_factor:.1f}"
        if self.ball_possesion["team"]:
            info_text += f" | Piłka: {self.ball_possesion['team']}{self.ball_possesion['player_id']}"
        info_label = self.font.render(info_text, True, BLACK, WHITE)
        self.screen.blit(info_label, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)  # max 60 FPS
    
    def set_time_factor(self, factor):
        """Ustawia mnożnik czasu symulacji"""
        self.time_factor = max(0.1, factor)  # Minimum 0.1, aby uniknąć zatrzymania
    
    def check_for_events(self):
        """Sprawdza zdarzenia pygame i reaguje na nie"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.set_time_factor(self.time_factor * 1.5)
                elif event.key == pygame.K_DOWN:
                    self.set_time_factor(self.time_factor / 1.5)
                elif event.key == pygame.K_SPACE:
                    # Reset pozycji piłki
                    self.ball_pos = torch.tensor([FIELD_WIDTH/2, FIELD_HEIGHT/2], 
                                               dtype=torch.float32, device=self.device)
                    self.ball_vel = torch.zeros(2, dtype=torch.float32, device=self.device)
        return True

class SimpleAgent:
    """Prosty agent do sterowania zawodnikiem"""
    def __init__(self, team, player_id, simulation):
        self.team = team
        self.player_id = player_id
        self.simulation = simulation
    
    def act(self):
        """Metoda decyzyjna agenta"""
        # Pobierz obserwacje
        observation = self.simulation.get_player_observation(self.team, self.player_id)
        
        # Pozycja piłki względem zawodnika
        ball_rel_pos = observation[0, :2].cpu().numpy()
        dist_to_ball = np.linalg.norm(ball_rel_pos)
        
        # Decyzja o ruchu w kierunku piłki
        if dist_to_ball > 0.1:
            direction = ball_rel_pos / dist_to_ball
        else:
            direction = np.array([0, 0])
        
        # Decyzja o kopnięciu piłki
        kick = False
        kick_power = 0.0
        kick_direction_x = 0.0
        kick_direction_y = 0.0
        
        if dist_to_ball < CONTROL_DISTANCE:
            kick = True
            kick_power = MAX_KICK_POWER * 0.7
            
            # Kierunek kopnięcia w stronę bramki przeciwnika
            if self.team == "A":
                kick_direction_x = 1.0  # Prawo
                # Losowe odchylenie w pionie
                kick_direction_y = random.uniform(-0.3, 0.3)
            else:
                kick_direction_x = -1.0  # Lewo
                # Losowe odchylenie w pionie
                kick_direction_y = random.uniform(-0.3, 0.3)
        
        # Ustaw akcję w symulacji
        self.simulation.set_player_action(
            self.team, self.player_id,
            direction[0], direction[1],
            kick=kick, kick_power=kick_power,
            kick_direction_x=kick_direction_x, kick_direction_y=kick_direction_y
        )

def main():
    """Główna funkcja demonstracyjna"""
    # Inicjalizacja symulacji z 5 zawodnikami na drużynę
    sim = FootballSimulation(num_players_per_team=5)
    
    # Utwórz agentów dla drużyny A
    team_a_agents = [SimpleAgent("A", i, sim) for i in range(5)]
    
    # Utwórz agentów dla drużyny B
    team_b_agents = [SimpleAgent("B", i, sim) for i in range(5)]
    
    # Główna pętla symulacji
    running = True
    while running:
        # Sprawdź zdarzenia
        running = sim.check_for_events()
        
        # Decyzje agentów
        for agent in team_a_agents + team_b_agents:
            agent.act()
        
        # Krok aktualizacji symulacji
        goal_scored, goal_team = sim.update()
        
        # Jeśli zdobyto gola, wyświetl informację
        if goal_scored:
            print(f"GOL! Drużyna {goal_team} zdobyła bramkę! Wynik: A {sim.score_a} - {sim.score_b} B")
        
        # Renderowanie
        sim.render()
    
    pygame.quit()

if __name__ == "__main__":
    main()