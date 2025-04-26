import multiprocessing
import socket
import time

class RCSSClient:
    def __init__(self, team_name="TeamPython", player_num=1, is_goalie=False, server_host="localhost", server_port=6000):
        self.team_name = team_name
        self.player_num = player_num
        self.is_goalie = is_goalie
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
            
            # Add goalie designation if this is a goalie
            init_msg = f"(init {self.team_name} (version 19){' (goalie)' if self.is_goalie else ''})"
            self.socket.sendto(init_msg.encode(), (self.server_host, self.server_port))
            
            data, addr = self.socket.recvfrom(4096)
            response = data.decode()
            
            if "(init" in response:
                print(f"[{self.team_name}] {'Goalie' if self.is_goalie else 'Player'} połączony z serwerem.")
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

    def catch(self, direction):
        """Próba złapania piłki przez bramkarza."""
        if not self.is_connected:
            return "Nie połączono z serwerem"
        if not self.is_goalie:
            return "Tylko bramkarz może łapać piłkę"
            
        catch_msg = f"(catch {direction})"
        self.socket.sendto(catch_msg.encode(), (self.server_host, self.server_port))
        return f"Próba złapania piłki w kierunku {direction}"
    
    def kick(self, power, direction):
        """Kick the ball with specified power and direction."""
        if not self.is_connected:
            return "Not connected to server"
            
        kick_msg = f"(kick {power} {direction})"
        self.socket.sendto(kick_msg.encode(), (self.server_host, self.server_port))
        return f"Kicked ball with power {power} and direction {direction}"
        
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
        
        resp_queue.put({"status": "connected", "player_num": self.player_num, "is_goalie": self.is_goalie})
        
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
                    elif cmd["action"] == "catch" and self.is_goalie:
                        direction = cmd.get("direction", 0)
                        result = self.catch(direction)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "kick":
                        power = cmd.get("power", 10)
                        direction = cmd.get("direction", 0)
                        result = self.kick(power, direction)
                        resp_queue.put({"status": "success", "message": result})
                    elif cmd["action"] == "exit":
                        break
                except multiprocessing.queues.Empty:
                    pass  
                                
                time.sleep(0.01)  
                
        except Exception as e:
            resp_queue.put({"status": "error", "message": str(e)})
        finally:
            self.disconnect()