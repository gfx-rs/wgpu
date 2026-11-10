import hashlib
import http.server
import importlib.util
import json
import pathlib
import re
import shlex
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("download", ROOT / "download-release" / "download.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARCHIVE = b"verified release archive"
DIGEST = hashlib.sha256(ARCHIVE).hexdigest()


class Downloads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = pathlib.Path(self.temp.name)

    def download(self):
        return MODULE.download("owner/project", "v1.2.3", "asset.zip", DIGEST, self.directory)

    def success(self, args, **kwargs):
        self.assertEqual(args[:5], ["gh", "release", "download", "v1.2.3", "--repo"])
        self.assertEqual(args[5], "owner/project")
        self.assertEqual(args[args.index("--pattern") + 1], "asset.zip")
        self.assertEqual(kwargs, {"check": True, "timeout": 120})
        pathlib.Path(args[args.index("--output") + 1]).write_bytes(ARCHIVE)

    def test_download_verifies_before_publishing(self):
        with patch.object(MODULE.subprocess, "run", side_effect=self.success):
            result = self.download()
        self.assertEqual(result.read_bytes(), ARCHIVE)
        self.assertEqual(list(self.directory.iterdir()), [result])

    def test_verified_cache_hit_avoids_network(self):
        (self.directory / "asset.zip").write_bytes(ARCHIVE)
        with patch.object(MODULE.subprocess, "run") as command:
            self.download()
            command.assert_not_called()

    def test_tainted_cache_is_rejected(self):
        (self.directory / "asset.zip").write_bytes(b"tainted")
        with patch.object(MODULE.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.download()
            command.assert_not_called()

    def test_http_error_payload_is_not_published(self):
        def error_page(args, **kwargs):
            pathlib.Path(args[args.index("--output") + 1]).write_bytes(b"HTTP error page")
        with patch.object(MODULE.subprocess, "run", side_effect=error_page):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                self.download()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_transient_failure_retries(self):
        attempts = 0
        def transient(args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                pathlib.Path(args[args.index("--output") + 1]).write_bytes(b"partial")
                raise subprocess.CalledProcessError(1, args, stderr="HTTP 503")
            return self.success(args, **kwargs)
        with patch.object(MODULE.subprocess, "run", side_effect=transient), patch.object(MODULE.time, "sleep") as sleep:
            self.download()
            self.assertEqual([call.args for call in sleep.call_args_list], [(1,), (2,)])
        self.assertEqual(attempts, 3)
        self.assertEqual(len(list(self.directory.iterdir())), 1)

    def test_repeated_failure_is_bounded(self):
        with patch.object(MODULE.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "gh")) as command, patch.object(MODULE.time, "sleep"):
            with self.assertRaises(subprocess.CalledProcessError):
                self.download()
            self.assertEqual(command.call_count, 4)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_timeout_retries(self):
        calls = [subprocess.TimeoutExpired("gh", 120), None]
        def timeout(args, **kwargs):
            result = calls.pop(0)
            if result:
                raise result
            return self.success(args, **kwargs)
        with patch.object(MODULE.subprocess, "run", side_effect=timeout), patch.object(MODULE.time, "sleep"):
            self.download()
        self.assertEqual(calls, [])

    def test_unsafe_filenames_are_rejected(self):
        for filename in ["../asset.zip", "..", "x\\asset.zip", "*.zip", "asset?.zip", ""]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                MODULE.download("owner/project", "v1", filename, DIGEST, self.directory)

    def test_digest_is_required(self):
        for digest in ["", "sha256:" + DIGEST, "0" * 63, "x" * 64]:
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                MODULE.download("owner/project", "v1", "asset.zip", digest, self.directory)

    def test_cache_key_has_no_cross_version_fallback(self):
        action = (ROOT / "download-release" / "action.yml").read_text()
        for field in ["runner.os", "runner.arch", "inputs.repository", "inputs.tag", "inputs.filename", "inputs.sha256"]:
            self.assertIn("${{ " + field + " }}", next(line for line in action.splitlines() if "key: wgpu-release-" in line))
        self.assertNotIn("restore-keys:", action)
        self.assertIn("GH_TOKEN: ${{ github.token }}", action)

    def test_only_trusted_default_branch_saves_before_tests(self):
        action = (ROOT / "download-release" / "action.yml").read_text()
        self.assertIn("github.ref == format('refs/heads/{0}', github.event.repository.default_branch) &&", action)
        self.assertIn("contains(fromJSON('[\"push\", \"schedule\", \"workflow_dispatch\"]'), github.event_name)", action)
        self.assertIn("actions/cache/restore@", action)
        self.assertIn("actions/cache/save@", action)
        self.assertNotIn("save-always:", action)
        self.assertLess(action.index("python \"$ACTION_PATH/download.py\""), action.index("- name: Save verified"))

    def test_queue_retry_preserves_routes_and_merge_methods(self):
        config = (ROOT.parent / "mergify.yml").read_text()
        self.assertEqual(config.count("max_checks_retries: 1"), 2)
        self.assertIn("merge_method: rebase", config)
        self.assertIn("merge_method: squash", config)
        self.assertIn("'label = \"merge: rebase\"'", config)
        self.assertIn("'-label = \"merge: rebase\"'", config)
        self.assertNotIn("branch_protection_injection_mode: none", config)


class CurlDownloads(unittest.TestCase):
    def commands(self):
        repo = ROOT.parent.parent
        commands = []
        for path in [ROOT / "install-vulkan-sdk/action.yml", repo / ".github/workflows/lazy.yml"]:
            for line in path.read_text().splitlines():
                line = line.strip().removeprefix("run: ")
                if line.startswith(("curl ", "curl.exe ")):
                    args = shlex.split(line)
                    args[0] = "curl"
                    commands.append(args)
        for filename in ["install_warp.rs", "install_agility_sdk.rs"]:
            source = (repo / "xtask/src" / filename).read_text().split('.cmd("curl.exe")', 1)[1]
            arguments = re.search(r"\.args\(\[(.*?)\]\)", source, re.S).group(1)
            args = [json.loads(token) if token.startswith('"') else "https://fixture.invalid/archive" for token in re.findall(r'"[^"\\]*"|&(?:warp_url|url)', arguments)]
            commands.append(["curl", *args])
        self.assertEqual(len(commands), 6)
        for args in commands:
            for flag in ["--fail", "--show-error", "-L"]:
                self.assertIn(flag, args)
            for flag, value in [("--retry", "5"), ("--retry-max-time", "300"), ("--connect-timeout", "30"), ("--max-time", "120")]:
                self.assertEqual(args[args.index(flag) + 1], value)
        return commands

    def check_http(self, statuses, success):
        class Handler(http.server.BaseHTTPRequestHandler):
            requests = 0
            def do_GET(self):
                status = statuses[min(type(self).requests, len(statuses) - 1)]
                type(self).requests += 1
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"archive" if status == 200 else b"error page")
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for args in self.commands():
                Handler.requests = 0
                with tempfile.TemporaryDirectory() as temp:
                    output = pathlib.Path(temp) / "archive"
                    extracted = pathlib.Path(temp) / "extracted"
                    url = next(i for i, arg in enumerate(args) if arg.startswith("https://"))
                    args[url] = f"http://127.0.0.1:{server.server_port}/archive"
                    args[args.index("-o") + 1] = str(output)
                    args += ["--retry", "2", "--retry-max-time", "5"]
                    script = shlex.join(args) + "; touch " + shlex.quote(str(extracted))
                    result = subprocess.run(["bash", "-e", "-c", script], capture_output=True, timeout=10)
                    self.assertEqual(result.returncode == 0, success, result.stderr)
                    self.assertEqual(extracted.exists(), success)
                    if success:
                        self.assertEqual(output.read_bytes(), b"archive")
                        self.assertEqual(Handler.requests, 3)
        finally:
            server.shutdown()
            server.server_close()

    def test_transient_errors_retry(self):
        self.check_http([503, 503, 200], True)

    def test_http_error_prevents_extraction(self):
        self.check_http([404], False)


if __name__ == "__main__":
    unittest.main()
