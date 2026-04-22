import subprocess
import sys
import os

# Clear proxy env vars
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    if var in os.environ:
        del os.environ[var]

# Set no_proxy
os.environ["NO_PROXY"] = "*"

# Install using subprocess with clean env
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "-i",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
    ],
    capture_output=True,
    text=True,
    env=os.environ,
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)
