"""Minimal Docker runner: one fresh container per run, removed afterwards."""

from __future__ import annotations

import io
import tarfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import docker  # type: ignore
from docker.errors import ImageNotFound, NotFound  # type: ignore

GOLD_APPLY = (
    "git apply --verbose {p} || "
    "git apply --verbose --reject {p} || "
    "patch --batch --fuzz=5 -p1 -i {p}"
)


EVAL_MARK = ">>>>> Eval script\n"  # judge only scans what follows this for test-patch errors


@dataclass
class RunResult:
    ok: bool
    log: str
    stage: str  # where it stopped: "image" | "<label>_apply" | "timeout" | "done"
    seconds: float


class DockerRunner:
    def __init__(self, *, mem_limit: str = "8g", cpus: float = 2.0, network: bool = True,
                 platform: str | None = "linux/amd64", pull: bool = True, workers: int = 4):
        # one client shared by all worker threads; size its connection pool for them
        self.client = docker.from_env(timeout=600, max_pool_size=max(10, 2 * workers + 4))
        self.mem_limit, self.cpus, self.network = mem_limit, cpus, network
        self.platform, self.pull = platform, pull
        self._pull_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._guard = threading.Lock()

    def ensure_image(self, image: str) -> str | None:
        """Pull once per image even when several workers need it at the same time."""
        with self._guard:
            lock = self._pull_locks[image]
        with lock:
            try:
                self.client.images.get(image)
                return None
            except ImageNotFound:
                if not self.pull:
                    return f"image {image} not present and --no-pull set"
            try:
                self.client.images.pull(image, platform=self.platform)
                return None
            except Exception as e:  # noqa: BLE001
                return f"pull failed for {image}: {e}"

    @staticmethod
    def _put(container, path: str, content: str) -> None:
        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.rsplit("/", 1)[-1])
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        container.put_archive(path.rsplit("/", 1)[0] or "/", buf.getvalue())

    def run(self, image: str, script: str, *, workdir: str, patches: list[tuple[str, str]] | None = None,
            timeout_s: int = 1800) -> RunResult:
        """Fresh container: apply `patches` in order ([(label, diff)], e.g. setup bug then gold fix),
        then run the eval script. Stops with stage=<label>_apply if a patch does not apply."""
        t0 = time.time()
        err = self.ensure_image(image)
        if err:
            return RunResult(False, err, "image", time.time() - t0)
        container = self.client.containers.run(
            image, command="sleep infinity", detach=True,
            network_disabled=not self.network, mem_limit=self.mem_limit,
            nano_cpus=int(self.cpus * 1e9), platform=self.platform,
        )
        try:
            log_parts: list[str] = []
            for k, (label, diff) in enumerate(patches or []):
                path = f"/tmp/p{k}_{label}.patch"
                self._put(container, path, diff if diff.endswith("\n") else diff + "\n")
                cmd = f"cd {workdir} && " + GOLD_APPLY.format(p=path)
                r = container.exec_run(["/bin/bash", "-c", cmd], demux=False)
                out = (r.output or b"").decode("utf-8", "replace")
                log_parts.append(f">>>>> Apply {label}\n" + out)
                if r.exit_code != 0:
                    return RunResult(False, "\n".join(log_parts), f"{label}_apply", time.time() - t0)
            self._put(container, "/tmp/eval.sh", script)
            r = container.exec_run(
                ["/bin/bash", "-c", f"timeout -k 10 {timeout_s} /bin/bash /tmp/eval.sh 2>&1"], demux=False
            )
            out = (r.output or b"").decode("utf-8", "replace")
            log_parts.append(EVAL_MARK + out)
            stage = "timeout" if r.exit_code == 124 else "done"
            return RunResult(stage == "done", "\n".join(log_parts), stage, time.time() - t0)
        finally:
            try:
                container.remove(force=True)
            except NotFound:
                pass

    def remove_image(self, image: str) -> None:
        try:
            self.client.images.remove(image, force=True)
        except Exception:  # noqa: BLE001
            pass
