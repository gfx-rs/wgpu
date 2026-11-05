import hashlib
import os
import pathlib
import re
import subprocess
import tempfile
import time


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def download(repository, tag, filename, expected, directory):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repository):
        raise ValueError("Invalid release repository")
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Expected a pinned SHA-256 digest")
    if pathlib.Path(filename).name != filename or any(c in filename for c in "\\*?[]") or filename in ("", ".", ".."):
        raise ValueError("Expected an exact release asset filename")
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists():
        verify(destination, expected)
        print(f"Verified cached {filename}")
        return destination
    for attempt in range(4):
        with tempfile.NamedTemporaryFile(dir=directory, prefix="download-", delete=False) as temporary:
            partial = pathlib.Path(temporary.name)
        try:
            subprocess.run(
                ["gh", "release", "download", tag, "--repo", repository, "--pattern", filename, "--output", str(partial), "--clobber"],
                check=True,
                timeout=120,
            )
            verify(partial, expected)
            partial.replace(destination)
            print(f"Verified downloaded {filename}")
            return destination
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
        finally:
            partial.unlink(missing_ok=True)
    raise AssertionError("Download retry loop exhausted")


if __name__ == "__main__":
    download(
        os.environ["RELEASE_REPOSITORY"],
        os.environ["RELEASE_TAG"],
        os.environ["RELEASE_FILENAME"],
        os.environ["RELEASE_SHA256"],
        os.environ["RELEASE_DIRECTORY"],
    )
