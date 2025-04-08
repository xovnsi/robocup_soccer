import socket
import time

SERVER_ADDR = ('127.0.0.1', 6000)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    # Initialize the agent
    init_msg = "(init SimpleTeam (version 15))"
    sock.sendto(init_msg.encode(), SERVER_ADDR)
    sock.recvfrom(4096)  # Wait for response

    print("[Agent] Initialized. Starting to move.")

    # Start moving forward
    while True:
        sock.sendto("(dash 80)".encode(), SERVER_ADDR)  # Dash command
        time.sleep(1)  # Dash every second

if __name__ == "__main__":
    main()
