import subprocess

# Launch 5 agents
for i in range(5):
    subprocess.Popen(["python3", "base_agent.py"])
