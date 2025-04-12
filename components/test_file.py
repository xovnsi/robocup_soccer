import socket
import time

ADDR = ('127.0.0.1', 6000)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    # Initialize the agent
    print("Initializing agent...")
    sock.sendto(b'(init SimpleTeam (version 15))', ADDR)
    time.sleep(0.2)  

    print("Moving to position (10, 10)...")
    sock.sendto(b' (move 10 10)', ADDR) 

    time.sleep(2)

    while True:
        try:
            # Receive messages from the server
            data, _ = sock.recvfrom(4096)
            msg = data.decode()

            # print("Got SEE:", msg) 

            # # Check for ball presence
            # if "(ball)" in msg:
            #     print("Ball spotted! Kicking...")
            #     sock.sendto(b'(kick 100 0)', ADDR) 
            #     break


        except socket.timeout:
            print("Waiting for response...")
            continue

        time.sleep(0.1)

if __name__ == "__main__":
    main()
