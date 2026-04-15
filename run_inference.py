import os
import sys
import signal
import platform
import argparse
import subprocess

QEMU_PATH = "/home/momo/simulator/bin/qemu-riscv64"
RISCV_SYSROOT = "/home/momo/riscv-gnu-toolchain-cx1c/sysroot-20250410-1"

def run_command(command, shell=False):
    try:
        print("Running:", " ".join(command) if isinstance(command, list) else command)
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    if not os.path.exists(main_path):
        print(f"Executable not found: {main_path}")
        sys.exit(1)

    command = [
        QEMU_PATH,
        "-L", RISCV_SYSROOT,
        main_path,
        "-m", args.model,
        "-n", str(args.n_predict),
        "-t", str(args.threads),
        "-p", args.prompt,
        "-ngl", "0",
        "-c", str(args.ctx_size),
        "--temp", str(args.temperature),
        "-b", "1",
    ]

    if args.conversation:
        command.append("-cnv")

    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("-m", "--model", type=str, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, default=128)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-t", "--threads", type=int, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, default=0.8)
    parser.add_argument("-cnv", "--conversation", action="store_true")

    global args
    args = parser.parse_args()
    run_inference()