import subprocess
import sys
import os
import urllib.request
import urllib.parse

# Clear all proxy-related environment variables first
proxy_vars = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
]
for var in proxy_vars:
    os.environ.pop(var, None)

# Create a completely clean environment for subprocess
clean_env = os.environ.copy()
for var in proxy_vars:
    clean_env.pop(var, None)
clean_env["NO_PROXY"] = "*"
clean_env["no_proxy"] = "*"

# Test urllib directly
print("Testing urllib.request directly...")
try:
    req = urllib.request.Request(
        "https://pypi.tuna.tsinghua.edu.cn/simple/torch/",
        headers={"User-Agent": "pip/25.0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        print(f"urllib works: {response.status}")
except Exception as e:
    print(f"urllib failed: {e}")

# Try pip with subprocess and clean env
print("\n\nTrying pip with subprocess and clean env...")
cmd = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "torch",
    "-i",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "--timeout",
    "60",
]

print(f"Command: {' '.join(cmd)}")
print(f"Proxy env vars: {[k for k in clean_env if 'proxy' in k.lower()]}")

result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=clean_env)

print("\nSTDOUT:")
print(result.stdout[:2000])  # Limit output

print("\nSTDERR:")
print(result.stderr[:2000])  # Limit output

print(f"\nReturn code: {result.returncode}")
