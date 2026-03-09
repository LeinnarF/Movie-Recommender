import json
import os
import subprocess
import sys
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


class TerminalState:
    def __init__(self):
        self.cwd = os.getcwd()
        self.lock = threading.Lock()


STATE = TerminalState()


class DashboardHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/terminal/exec":
            self._send_json({"error": "Not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)

        try:
            data = json.loads(raw.decode("utf-8"))
            command = (data.get("command") or "").strip()
        except Exception:
            self._send_json({"error": "Invalid JSON payload"}, status=400)
            return

        if not command:
            self._send_json({"error": "Command is required"}, status=400)
            return

        with STATE.lock:
            cwd = STATE.cwd

            # Handle `cd` as a shell state update so cwd persists across commands.
            if command.lower().startswith("cd ") or command.lower() == "cd":
                target = command[2:].strip() if command.lower().startswith("cd ") else ""
                if target:
                    target = target.strip('"')
                    new_cwd = target if os.path.isabs(target) else os.path.join(cwd, target)
                    new_cwd = os.path.abspath(new_cwd)
                else:
                    new_cwd = os.path.expanduser("~")

                if os.path.isdir(new_cwd):
                    STATE.cwd = new_cwd
                    self._send_json(
                        {
                            "stdout": "",
                            "stderr": "",
                            "exit_code": 0,
                            "cwd": STATE.cwd,
                        }
                    )
                else:
                    self._send_json(
                        {
                            "stdout": "",
                            "stderr": f"The system cannot find the path specified: {new_cwd}",
                            "exit_code": 1,
                            "cwd": STATE.cwd,
                        }
                    )
                return

            shell = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]

            try:
                proc = subprocess.run(
                    shell,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    encoding="utf-8",
                    errors="replace",
                )
                self._send_json(
                    {
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "exit_code": proc.returncode,
                        "cwd": STATE.cwd,
                    }
                )
            except subprocess.TimeoutExpired:
                self._send_json(
                    {
                        "stdout": "",
                        "stderr": "Command timed out after 120 seconds.",
                        "exit_code": 124,
                        "cwd": STATE.cwd,
                    }
                )


def main():
    host = "127.0.0.1"
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port. Usage: python web_terminal.py [port]")
            raise SystemExit(1)

    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard server running at http://{host}:{port}/dashboard.html")
    print("Use Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
