"""Start the dashboard and the IEX session, and stop both together.

This is a foreground launcher. It does not register a Windows service,
a startup task, or a scheduler. Closing the Start Portfolio window runs
the console close handler, which ends both child processes.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from pathlib import Path


def launch_commands(python: Path, root: Path) -> tuple[list[str], list[str]]:
    """The data process and the Streamlit process. Neither is a full-year download."""
    data = [str(python), "-m", "data_collection.live_session"]
    dashboard = [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(root / "app" / "dashboard.py"),
        "--browser.gatherUsageStats",
        "false",
    ]
    return data, dashboard


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """End each process and the children it started."""
    for proc in processes:
        if proc.poll() is not None:
            continue
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def install_close_handler(callback) -> None:
    """On Windows, closing the console window calls ``callback``."""
    if sys.platform != "win32":
        return
    handler_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)

    def _handler(ctrl_type: int) -> int:
        # 2 is CTRL_CLOSE_EVENT, 5 is logoff, 6 is shutdown.
        if ctrl_type in (2, 5, 6):
            callback()
            return 1
        return 0

    handle = handler_type(_handler)
    if not ctypes.windll.kernel32.SetConsoleCtrlHandler(handle, True):
        raise ctypes.WinError()
    install_close_handler.keep = handle  # type: ignore[attr-defined]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    data_command, dashboard_command = launch_commands(python, root)
    data = subprocess.Popen(data_command, cwd=root)
    dashboard = subprocess.Popen(dashboard_command, cwd=root)
    processes = [data, dashboard]
    print("Portfolio is running.")
    print(f"Market data process: {data.pid}")
    print(f"Dashboard process: {dashboard.pid}")
    print("Close this window to stop the dashboard and market-data updates.")

    def shutdown() -> None:
        stop_processes(processes)

    install_close_handler(shutdown)
    try:
        while any(proc.poll() is None for proc in processes):
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
    print("Portfolio stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
