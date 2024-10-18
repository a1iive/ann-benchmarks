import subprocess

# 要处理的容器列表
containers = ["0f4aa91680be", "0486d9789e93", "b1eebee8f8a7"]
def send_signal(container_name):
    """向指定的容器发送 SIGUSR1 信号"""
    subprocess.run(["docker", "kill", "-s", "SIGUSR2", container_name])

def main(): 
    for container in containers:
        send_signal(container)

if __name__ == "__main__":
    main()
