"""Local stand-in for the Vercel deploy: serves public/ and routes POST /api/trial to api/trial.py."""
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from api.trial import handler as TrialHandler  # noqa: E402


class Dev(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "public"), **k)

    def do_POST(self):
        if self.path.startswith("/api/trial"):
            return TrialHandler.do_POST(self)
        self.send_error(404)

    _send = TrialHandler._send


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Dev).serve_forever()
