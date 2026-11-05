import http.server
import pathlib
import shlex
import subprocess
import tempfile
import threading
import unittest

ROOT = pathlib.Path(__file__).resolve().parent


class Downloads(unittest.TestCase):
    def test_transient_http_error_retries(self):
        self.check_download([503, 503, 200], True)

    def test_http_error_does_not_extract(self):
        self.check_download([404], False)

    def check_download(self, statuses, success):
        class Handler(http.server.BaseHTTPRequestHandler):
            requests = 0

            def do_GET(self):
                status = statuses[min(self.requests, len(statuses) - 1)]
                type(self).requests += 1
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"archive" if status == 200 else b"error page")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for action in ["install-mesa", "install-dxc"]:
                text = (ROOT / action / "action.yml").read_text()
                commands = [line.strip() for line in text.splitlines() if line.strip().startswith(("curl ", "curl.exe "))]
                for command in commands:
                    args = shlex.split(command)
                    self.assertIn("--fail", args)
                    self.assertEqual(args[args.index("--retry") + 1], "8")
                    self.assertEqual(args[args.index("--retry-max-time") + 1], "300")
                    Handler.requests = 0
                    with tempfile.TemporaryDirectory() as temp:
                        download = pathlib.Path(temp) / "archive"
                        extracted = pathlib.Path(temp) / "extracted"
                        url = next(i for i, arg in enumerate(args) if arg.startswith("https://"))
                        args[0] = "curl"
                        args[url] = f"http://127.0.0.1:{server.server_port}/archive"
                        args[args.index("-o") + 1] = str(download)
                        args += ["--retry", "2", "--retry-max-time", "5"]
                        script = shlex.join(args) + "; touch " + shlex.quote(str(extracted))
                        result = subprocess.run(["bash", "-e", "-c", script], capture_output=True, timeout=10)
                        self.assertEqual(result.returncode == 0, success, result.stderr)
                        self.assertEqual(extracted.exists(), success)
                        if success:
                            self.assertEqual(download.read_bytes(), b"archive")
                            self.assertEqual(Handler.requests, 3)
        finally:
            server.shutdown()
            server.server_close()

    def test_queue_retry_preserves_routes_and_merge_methods(self):
        config = (ROOT.parent / "mergify.yml").read_text()
        self.assertEqual(config.count("max_checks_retries: 1"), 2)
        self.assertIn("merge_method: rebase", config)
        self.assertIn("merge_method: squash", config)
        self.assertIn("'label = \"merge: rebase\"'", config)
        self.assertIn("'-label = \"merge: rebase\"'", config)
        self.assertNotIn("branch_protection_injection_mode: none", config)


if __name__ == "__main__":
    unittest.main()
