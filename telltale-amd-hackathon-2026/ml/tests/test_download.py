"""The downloader must survive flaky Wi-Fi: detect truncated transfers, resume with HTTP Range, verify zips."""

import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from telltale_ml.acoustic.data import Clip
from telltale_ml.datasets.download import _download, _extract, _spread_over_sections

PAYLOAD = os.urandom(300_000)


class FlakyHandler(BaseHTTPRequestHandler):
    """Serves PAYLOAD; the first full (non-Range) request is cut off halfway, like a dropped connection."""

    calls = 0

    def log_message(self, *args):  # keep test output clean
        pass

    def do_GET(self):
        FlakyHandler.calls += 1
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not rng and FlakyHandler.calls == 1:
            self.wfile.write(body[: len(body) // 2])  # drop the connection mid-transfer
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


@pytest.fixture()
def server():
    FlakyHandler.calls = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.zip"
    httpd.shutdown()


def test_download_resumes_after_a_dropped_connection(server, tmp_path):
    dest = tmp_path / "file.zip"
    _download(server, dest, timeout=10, retries=5)
    assert dest.read_bytes() == PAYLOAD  # complete and byte-identical
    assert FlakyHandler.calls >= 2  # needed a resumed request
    assert not dest.with_suffix(".zip.part").exists()


def test_damaged_zip_is_deleted_with_clear_message(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(RuntimeError, match="damaged"):
        _extract(bad, tmp_path / "out")
    assert not bad.exists()


def test_extract_runs_once(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x/hello.txt", "hi")
    _extract(archive, tmp_path / "out")
    (tmp_path / "out" / "x" / "hello.txt").write_text("edited")
    _extract(archive, tmp_path / "out")  # second call is skipped (marker), so the edit survives
    assert (tmp_path / "out" / "x" / "hello.txt").read_text() == "edited"


def test_demo_clips_spread_over_sections_and_prefer_source(tmp_path):
    clips = [
        Clip(tmp_path / f"{s}_{d}_{i}.wav", "anomaly", "test", s, d)
        for s in ("section_00", "section_01", "section_02")
        for d in ("source", "target")
        for i in range(4)
    ]
    picked = _spread_over_sections(clips, 3)
    assert [c.group for c in picked] == ["section_00", "section_01", "section_02"]
    assert all(c.domain == "source" for c in picked)
    assert len(_spread_over_sections(clips[:2], 3)) == 2  # fewer clips than asked: take what exists
