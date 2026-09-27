import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from placebo import service  # noqa: E402

MAX_BODY = 1_200_000


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "the request is larger than 1 MB"})
            req = json.loads(self.rfile.read(n) or b"{}")
            q, csv_text = req.get("question"), req.get("csv")
            if not isinstance(q, str) or (csv_text is not None and not isinstance(csv_text, str)):
                return self._send(400, {"error": "question must be text"})
            return self._send(200, service.run(q, csv_text))
        except service.UserError as e:
            return self._send(400, {"error": str(e)})
        except json.JSONDecodeError:
            return self._send(400, {"error": "the request was not valid JSON"})
        except Exception as e:
            return self._send(500, {"error": f"server error ({type(e).__name__})"})
