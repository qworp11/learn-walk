import socket, sys
ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(3)
s.sendto(b"ping", (ip, 7777))
try:
    data, addr = s.recvfrom(64)
    print("REPLY:", data, "from", addr)
except socket.timeout:
    print("TIMEOUT:", ip)
