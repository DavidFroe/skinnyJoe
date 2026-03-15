import pty
import subprocess
import os
import time
import select

master_fd, slave_fd = pty.openpty()
env = os.environ.copy()
env["TERM"] = "dumb"

proc = subprocess.Popen(
    ["lms", "chat", "lama3.2-1b-translatev4"],
    stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
    env=env
)
os.close(slave_fd)

# Read init
time.sleep(1.5)
init = os.read(master_fd, 8192)
print("--- INIT ---")
print(init.decode('utf-8', errors='replace'))

# Send prompt
os.write(master_fd, b"Hallo\n")
time.sleep(0.1)
os.write(master_fd, b"\r")
time.sleep(0.1)

# Read response until stable
print("--- RESPONSE ---")
buf = b""
last = time.time()
while time.time() - last < 2.0:
    r, _, _ = select.select([master_fd], [], [], 0.5)
    if r:
        try:
            chunk = os.read(master_fd, 8192)
            if chunk:
                buf += chunk
                last = time.time()
        except Exception:
            break

print(buf.decode('utf-8', errors='replace'))
proc.kill()
