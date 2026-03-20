#!/usr/bin/env python3
"""Mighty Giant — deploy & train on RunPod B200.

Usage:
    python deploy.py                              # create/get pod, sync code, setup
    python deploy.py --run "python3 train.py ..."  # sync + run command
    python deploy.py --setup                       # run setup_remote.sh on pod
    python deploy.py --stop                        # stop pod (preserves volume)
    python deploy.py --terminate                   # destroy pod entirely
    python deploy.py --status                      # show pod status
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── ANSI ──────────────────────────────────────────────────────────
B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
BLUE = "\033[38;5;75m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;221m"
RED = "\033[38;5;203m"
CYAN = "\033[38;5;117m"
MAGENTA = "\033[38;5;183m"
GRAY = "\033[38;5;245m"

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Config ────────────────────────────────────────────────────────
def load_env() -> dict[str, str]:
    env_file = SCRIPT_DIR / ".env"
    env = {}
    if not env_file.exists():
        return env
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


ENV = load_env()

RUNPOD_API_KEY = ENV.get("RUNPOD_API_KEY", "")
HF_API_KEY = ENV.get("HUGGING_FACE_API_KEY", "")
WANDB_API_KEY = ENV.get("WANDB_API_KEY", "")
WANDB_PROJECT = ENV.get("WANDB_PROJECT", "mighty-giant")
WANDB_ENTITY = ENV.get("WANDB_ENTITY", "")
SSH_PASSPHRASE = ENV.get("SSH_PASSPHRASE", "")

GPU_TYPE_ID = "NVIDIA B200"
GPU_FALLBACKS = ["NVIDIA H200 NVL", "NVIDIA H200", "NVIDIA H100 NVL"]
POD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME = "mighty-giant-train"
VOLUME_SIZE_GB = 2000
GPU_COUNT = 1

REMOTE_DIR = "/workspace/mighty-giant"
REMOTE_DATA = "/workspace/data/fineweb-edu/sample/350BT"
REMOTE_TOKENIZER = "/workspace/data/tokenizer-llama3"


# ── RunPod helpers ────────────────────────────────────────────────
def init_runpod():
    try:
        import runpod
    except ImportError:
        print(f"  {RED}✖{R} runpod not installed. Run: pip install runpod")
        sys.exit(1)
    runpod.api_key = RUNPOD_API_KEY
    return runpod


def find_pod(runpod):
    pods = runpod.get_pods()
    for pod in pods:
        if pod.get("name") == POD_NAME:
            return pod
    return None


def create_pod(runpod):
    print(f"  {CYAN}❯{R} {B}Creating pod{R} ({GPU_TYPE_ID})")

    pod_env = {
        "HF_TOKEN": HF_API_KEY,
        "HUGGING_FACE_HUB_TOKEN": HF_API_KEY,
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_ENTITY": WANDB_ENTITY,
    }

    gpu_ids = [GPU_TYPE_ID] + GPU_FALLBACKS
    for gpu_id in gpu_ids:
        try:
            pod = runpod.create_pod(
                name=POD_NAME,
                image_name=POD_IMAGE,
                gpu_type_id=gpu_id,
                gpu_count=GPU_COUNT,
                cloud_type="ALL",
                volume_in_gb=VOLUME_SIZE_GB,
                container_disk_in_gb=20,
                volume_mount_path="/workspace",
                ports="22/tcp",
                start_ssh=True,
                env=pod_env,
            )
            print(f"    {GREEN}✔{R} Created pod {B}{pod['id']}{R} with {gpu_id}")
            return pod
        except Exception as exc:
            if "no longer any instances available" in str(exc).lower():
                print(f"    {YELLOW}●{R} {gpu_id} not available, trying fallback...")
                continue
            raise

    # If all fallbacks fail, poll for primary GPU
    print(f"\n  {YELLOW}●{R} No GPUs available. Polling for {GPU_TYPE_ID}...")
    while True:
        try:
            pod = runpod.create_pod(
                name=POD_NAME,
                image_name=POD_IMAGE,
                gpu_type_id=GPU_TYPE_ID,
                gpu_count=GPU_COUNT,
                cloud_type="ALL",
                volume_in_gb=VOLUME_SIZE_GB,
                container_disk_in_gb=20,
                volume_mount_path="/workspace",
                ports="22/tcp",
                start_ssh=True,
                env=pod_env,
            )
            print(f"    {GREEN}✔{R} Created pod {B}{pod['id']}{R}")
            return pod
        except Exception:
            print(f"    {D}Retrying in 30s...{R}")
            time.sleep(30)


def wait_for_pod(runpod, pod_id: str, timeout: int = 600) -> dict:
    print(f"    {D}Waiting for pod to be ready...{R}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "")
        runtime = pod.get("runtime")
        if status == "RUNNING" and runtime and runtime.get("ports"):
            print(f"    {GREEN}✔{R} Pod is ready")
            return pod
        time.sleep(10)
    raise TimeoutError(f"Pod did not become ready within {timeout}s")


def get_ssh_details(pod: dict) -> tuple[str, int]:
    runtime = pod.get("runtime", {})
    ports = runtime.get("ports", [])
    for port_info in ports:
        if port_info.get("privatePort") == 22:
            ip = port_info.get("ip", "")
            public_port = port_info.get("publicPort", 22)
            return ip, int(public_port)
    pod_id = pod.get("id", "")
    return f"{pod_id}-ssh.proxy.runpod.net", 22


def ensure_pod(runpod) -> dict:
    pod = find_pod(runpod)
    if pod is not None:
        status = pod.get("desiredStatus", "")
        print(f"  {CYAN}❯{R} Found existing pod {B}{pod['id']}{R} (status={status})")
        if status == "EXITED":
            print(f"    {YELLOW}↻{R} Resuming stopped pod...")
            runpod.resume_pod(pod["id"], gpu_count=GPU_COUNT)
            pod = wait_for_pod(runpod, pod["id"])
        elif status != "RUNNING":
            pod = wait_for_pod(runpod, pod["id"])
        return pod
    return wait_for_pod(runpod, create_pod(runpod)["id"])


# ── SSH helpers ───────────────────────────────────────────────────
class SSHClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._askpass = None

        self._opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30",
        ]
        key_path = Path.home() / ".runpod" / "ssh" / "id_ed25519"
        if key_path.exists():
            self._opts.extend(["-i", str(key_path)])

        if SSH_PASSPHRASE:
            fd, path = tempfile.mkstemp(prefix="askpass_", suffix=".sh")
            with os.fdopen(fd, "w") as f:
                f.write(f"#!/bin/sh\necho '{SSH_PASSPHRASE}'\n")
            os.chmod(path, stat.S_IRWXU)
            self._askpass = path

    def _env(self) -> dict:
        env = os.environ.copy()
        if self._askpass:
            env["SSH_ASKPASS"] = self._askpass
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env.setdefault("DISPLAY", ":0")
        return env

    def _ssh_cmd(self) -> list[str]:
        return ["ssh"] + self._opts + ["-p", str(self.port), f"root@{self.host}"]

    def wait_for_ssh(self, timeout: int = 300):
        print(f"    {D}Waiting for SSH...{R}")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = subprocess.run(
                    self._ssh_cmd() + ["echo", "ok"],
                    capture_output=True, text=True, timeout=15, env=self._env(),
                )
                if result.returncode == 0:
                    print(f"    {GREEN}✔{R} SSH connected")
                    return
            except (subprocess.TimeoutExpired, OSError):
                pass
            time.sleep(5)
        raise TimeoutError("SSH not reachable")

    def run(self, cmd: str, timeout: int = 720) -> int:
        """Run command with live output. Returns exit code."""
        proc = subprocess.Popen(
            self._ssh_cmd() + [cmd],
            stdout=sys.stdout, stderr=sys.stderr,
            env=self._env(),
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return -1
        return proc.returncode

    def run_tty(self, cmd: str) -> int:
        """Run with TTY allocation (Ctrl+C propagates)."""
        full_cmd = ["ssh", "-tt"] + self._opts + ["-p", str(self.port), f"root@{self.host}", cmd]
        proc = subprocess.Popen(full_cmd, env=self._env())
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            return 130
        return proc.returncode

    def sync(self, local_dir: Path, remote_dir: str):
        """Sync code to pod via rsync."""
        ssh_cmd = f"ssh {' '.join(self._opts)} -p {self.port}"
        cmd = [
            "rsync", "-avz", "--delete",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.DS_Store",
            "--exclude=checkpoints/",
            "--exclude=Papers/",
            "--exclude=.env",
            "--exclude=.git/",
            "--exclude=.claude/",
            "-e", ssh_cmd,
            str(local_dir) + "/",
            f"root@{self.host}:{remote_dir}/",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=self._env())
        if result.returncode != 0:
            # Fallback to scp if rsync not available
            self.run(f"mkdir -p {remote_dir}")
            scp_opts = [o for o in self._opts] + ["-P", str(self.port)]
            scp_cmd = [
                "scp", "-r", *scp_opts,
                str(local_dir) + "/.",
                f"root@{self.host}:{remote_dir}/",
            ]
            result = subprocess.run(scp_cmd, capture_output=True, text=True, env=self._env())
            if result.returncode != 0:
                raise RuntimeError(f"Sync failed: {result.stderr}")
        return result

    def __del__(self):
        if self._askpass and Path(self._askpass).exists():
            Path(self._askpass).unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────
def print_header():
    print(f"""
  {B}{BLUE}╔══════════════════════════════════════╗{R}
  {B}{BLUE}║       Mighty Giant  {MAGENTA}Deploy{BLUE}           ║{R}
  {B}{BLUE}╚══════════════════════════════════════╝{R}
""")


def main():
    parser = argparse.ArgumentParser(description="Deploy Mighty Giant to RunPod")
    parser.add_argument("--run", type=str, default=None, help="Command to run on pod")
    parser.add_argument("--setup", action="store_true", help="Run setup_remote.sh on pod")
    parser.add_argument("--stop", action="store_true", help="Stop pod (preserves volume)")
    parser.add_argument("--terminate", action="store_true", help="Destroy pod entirely")
    parser.add_argument("--status", action="store_true", help="Show pod status")
    parser.add_argument("--train", action="store_true",
                        help="Shortcut: sync + run stage 1 training with wandb")
    args = parser.parse_args()

    print_header()

    rp = init_runpod()

    # Status check
    if args.status:
        pod = find_pod(rp)
        if pod is None:
            print(f"  {GRAY}No pod named '{POD_NAME}' found.{R}\n")
        else:
            status = pod.get("desiredStatus", "unknown")
            gpu = pod.get("machine", {}).get("gpuDisplayName", "unknown")
            color = GREEN if status == "RUNNING" else YELLOW if status == "EXITED" else RED
            print(f"  {GRAY}Pod:{R}    {B}{pod['id']}{R}")
            print(f"  {GRAY}Status:{R} {color}{status}{R}")
            print(f"  {GRAY}GPU:{R}    {gpu}")
            if status == "RUNNING":
                host, port = get_ssh_details(pod)
                print(f"  {GRAY}SSH:{R}    root@{host} -p {port}")
            print()
        return

    # Stop
    if args.stop:
        pod = find_pod(rp)
        if pod:
            rp.stop_pod(pod["id"])
            print(f"  {GREEN}✔{R} Pod {pod['id']} stopped (volume preserved)\n")
        else:
            print(f"  {GRAY}No pod to stop.{R}\n")
        return

    # Terminate
    if args.terminate:
        pod = find_pod(rp)
        if pod:
            rp.terminate_pod(pod["id"])
            print(f"  {GREEN}✔{R} Pod {pod['id']} terminated\n")
        else:
            print(f"  {GRAY}No pod to terminate.{R}\n")
        return

    # Ensure pod is running
    pod = ensure_pod(rp)
    host, port = get_ssh_details(pod)
    gpu = pod.get("machine", {}).get("gpuDisplayName", GPU_TYPE_ID)

    print(f"\n    {GRAY}Pod:{R}  {B}{pod['id']}{R}")
    print(f"    {GRAY}GPU:{R}  {gpu}")
    print(f"    {GRAY}SSH:{R}  root@{host} -p {port}")

    ssh = SSHClient(host, port)
    ssh.wait_for_ssh()

    # Sync code
    print(f"\n  {CYAN}❯{R} {B}Syncing code{R}")
    ssh.sync(SCRIPT_DIR, REMOTE_DIR)
    print(f"    {GREEN}✔{R} Code synced to {REMOTE_DIR}")

    # Setup
    if args.setup:
        print(f"\n  {CYAN}❯{R} {B}Running setup{R}")
        env_exports = (
            f"export WANDB_API_KEY='{WANDB_API_KEY}' && "
            f"export HF_TOKEN='{HF_API_KEY}' && "
            f"export HUGGING_FACE_HUB_TOKEN='{HF_API_KEY}' && "
        )
        # Use run_tty so it streams output and doesn't timeout during dataset download
        exit_code = ssh.run_tty(f"{env_exports} cd {REMOTE_DIR} && bash setup_remote.sh")
        if exit_code == 0:
            print(f"\n  {GREEN}✔{R} {B}Setup complete.{R}\n")
        else:
            print(f"\n  {RED}✖{R} {B}Setup failed (exit code {exit_code}).{R}\n")
        return

    # Train shortcut
    if args.train:
        args.run = (
            f"cd {REMOTE_DIR} && "
            f"WANDB_API_KEY='{WANDB_API_KEY}' "
            f"WANDB_PROJECT='{WANDB_PROJECT}' "
            f"WANDB_ENTITY='{WANDB_ENTITY}' "
            f"python3 train.py --stage 1 --wandb "
            f"--seq-len 256 --batch-size 32 "
            f"--dataset fineweb --dataset-path {REMOTE_DATA} "
            f"--tokenizer {REMOTE_TOKENIZER} "
            f"--checkpoint-dir /workspace/checkpoints"
        )

    # Run command
    if args.run:
        print(f"\n  {CYAN}❯{R} {B}Running on remote{R}")
        print(f"    {D}Ctrl+C will stop the remote process{R}")
        print(f"  {GRAY}──────────────────────────────────────{R}\n")

        # Inject env vars for wandb
        env_prefix = (
            f"export WANDB_API_KEY='{WANDB_API_KEY}' && "
            f"export WANDB_PROJECT='{WANDB_PROJECT}' && "
            f"export WANDB_ENTITY='{WANDB_ENTITY}' && "
        )
        exit_code = ssh.run_tty(f"{env_prefix} {args.run}")

        print(f"\n  {GRAY}──────────────────────────────────────{R}")
        if exit_code == 0:
            print(f"\n  {GREEN}✔{R} {B}Remote process completed.{R}\n")
        elif exit_code == 130:
            print(f"\n  {YELLOW}●{R} {B}Interrupted by user.{R}\n")
        else:
            print(f"\n  {RED}✖{R} {B}Exited with code {exit_code}.{R}\n")
        return

    # No command — just synced
    print(f"\n  {GREEN}✔{R} {B}Done.{R} Code synced, no command to run.\n")


if __name__ == "__main__":
    main()
