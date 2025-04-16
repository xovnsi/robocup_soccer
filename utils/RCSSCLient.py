# =============== Zastosowanie do RCSS ===============
import multiprocessing
import time
import socket
import re
import random
class RCSSClient:
    def __init__(self, team_name="TeamPython", player_num=1, server_host="localhost", server_port=6000):
        self.team_name = team_name
        self.player_num = player_num
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.is_connected = False
        self.position = (0, 0)
        
    def connect(self):
        """Połączenie z serwerem RCSS."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(0.5)
            
            init_msg = f"(init {self.team_name} (version 19))"
            self.socket.sendto(init_msg.encode(), (self.server_host, self.server_port))
            
            data, addr = self.socket.recvfrom(4096)
            response = data.decode()
            
            if "(init" in response:
                print(f"[{self.team_name}] Połączono z serwerem.")
                self.is_connected = True
                self.player_num = int(response.split()[2])
                self.server_port = addr[1]
                return True
            else:
                print(f"[{self.team_name}] Błąd połączenia: {response}")
                return False
                
        except Exception as e:
            print(f"[{self.team_name}] Błąd: {str(e)}")
            return False
    
    def move_to_position(self, x, y):
        """Przemieszczenie agenta na wskazaną pozycję."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        move_msg = f"(move {x} {y})"
        self.socket.sendto(move_msg.encode(), (self.server_host, self.server_port))
        self.position = (x, y)
        return f"Przemieszczono na pozycję ({x}, {y})"
    
    def dash(self, power, direction=0):
        """Wykonanie ruchu do przodu z określoną siłą."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        dash_msg = f"(dash {power} {direction})"
        self.socket.sendto(dash_msg.encode(), (self.server_host, self.server_port))
        return f"Wykonano dash z mocą {power} i kierunkiem {direction}"
    
    def turn(self, moment):
        """Obrót agenta o zadany kąt."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
            
        turn_msg = f"(turn {moment})"
        self.socket.sendto(turn_msg.encode(), (self.server_host, self.server_port))
        return f"Wykonano obrót o {moment}"
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.is_connected = False
            print(f"[{self.team_name}] Rozłączono.")
    
    def run(self, cmd_queue, resp_queue):
        """Główna pętla agenta."""
        if not self.connect():
            resp_queue.put({"status": "error", "message": "Nie udało się połączyć z serwerem"})
            return
        
        resp_queue.put({"status": "connected", "player_num": self.player_num})
        
        try:
            while True:
                # Sprawdź czy są nowe komendy
                try:
                    cmd = cmd_queue.get(block=False)
                    if cmd["action"] == "move":
                        x, y = cmd["position"]
                        result = self.move_to_position(x, y)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "dash":
                        power = cmd.get("power", 100)
                        direction = cmd.get("direction", 0)
                        result = self.dash(power, direction)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "turn":
                        moment = cmd.get("moment", 30)
                        result = self.turn(moment)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "exit":
                        break
                except multiprocessing.queues.Empty:
                    pass  # Brak nowych komend
                
                # Tutaj można dodać odbieranie i przetwarzanie komunikatów z serwera
                
                time.sleep(0.01)  # Krótkie opóźnienie
                
        except Exception as e:
            resp_queue.put({"status": "error", "message": str(e)})
        finally:
            self.disconnect()

def create_rcss_agent(team, player_num):
    """Tworzy i uruchamia agenta RCSS."""
    cmd_queue = multiprocessing.Queue()
    resp_queue = multiprocessing.Queue()
    
    client = RCSSClient(team_name=team, player_num=player_num)
    process = multiprocessing.Process(target=client.run, args=(cmd_queue, resp_queue))
    process.start()
    
    # Poczekaj na połączenie
    response = resp_queue.get(timeout=5)
    if response["status"] == "error":
        print(f"Błąd: {response['message']}")
        process.terminate()
        return None, None, None
    
    print(f"Agent {team} #{response['player_num']} połączony")
    
    return process, cmd_queue, resp_queue

def run_rcss_example():
    # Uruchom czterech agentów
    agents = []
    for i in range(1):
        proc, cmd_q, resp_q = create_rcss_agent("left", i+1)
        if proc:
            agents.append((proc, cmd_q, resp_q, f"left_{i+1}"))
    
    # Dodaj jednego agenta prawej drużyny
        proc, cmd_q, resp_q = create_rcss_agent("right", i+1)
        if proc:
            agents.append((proc, cmd_q, resp_q, f"right_{i+1}"))
    
    # Ustaw początkowe pozycje
    positions = [(random.randint(-20, 20), random.randint(-20, 20) )for _ in range(len(agents))]
    positions[0] = (0, 0)  
    for i, (_, cmd_q, resp_q, name) in enumerate(agents):
        cmd_q.put({"action": "move", "position": positions[i]})
        response = resp_q.get(timeout=1)
        print(f"{name}: {response['message']}")
    
    # Przykładowe sterowanie agentami
    time.sleep(5)
    
    # Niech pierwszy agent się porusza
    _, cmd_q, resp_q, name = agents[0]
    cmd_q.put({"action": "dash", "power": 100})
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    time.sleep(0.2)
    _, cmd_q, resp_q, name = agents[0]
    cmd_q.put({"action": "dash", "power": 100})
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    time.sleep(0.2)
    _, cmd_q, resp_q, name = agents[0]
    cmd_q.put({"action": "dash", "power": 100})
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    time.sleep(0.2)
    # Niech drugi agent się obróci
    _, cmd_q, resp_q, name = agents[1]
    cmd_q.put({"action": "turn", "moment": 90})
    response = resp_q.get(timeout=1)
    print(f"{name}: {response['message']}")
    
    # Zamknij wszystkie procesy
    """  time.sleep(155)  # Daj czas na obserwację
    for proc, cmd_q, _, _ in agents:
        cmd_q.put({"action": "exit"})
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()"""
    
if __name__ == "__main__":
    run_rcss_example()