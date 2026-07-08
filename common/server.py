"""Launch / stop a serened instance and sample its RAM + on-disk footprint.

Follows CLAUDE.md: `serened <datadir> --listen='postgres://0.0.0.0:<port>'`,
default database `postgres`, user `postgres`. The process is always killed and
(optionally) its scratch datadir removed on exit.
"""

import os
import shutil
import signal
import socket
import subprocess
import threading
import time

import psycopg

from . import sdb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_BINARY = os.path.join(REPO_ROOT, "build_bench", "bin", "serened")


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_status_kb(pid, field):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _tree_pids(root):
    """root pid plus all descendants (handles launcher->server processes like ES)."""
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            ppid = int(parts[1])  # field 4 (ppid) after comm
            children.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(children.get(p, ()))
    return out


def _tree_rss_kb(root):
    total = 0
    for p in _tree_pids(root):
        kb = _read_status_kb(p, "VmRSS")
        if kb:
            total += kb
    return total


class RssSampler(threading.Thread):
    """Samples summed VmRSS of a process tree (root + descendants), tracking a
    running peak and per-phase peaks. Tree-summing makes multi-process engines
    (e.g. an ES launcher that forks the server JVM) measure correctly; for a
    single-process server it is just that process's RSS."""

    def __init__(self, pid, interval=0.05):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self._stop_ev = threading.Event()
        self._lock = threading.Lock()
        self._phase = None
        self._phase_peak = {}
        self.peak_kb = 0

    def run(self):
        while not self._stop_ev.is_set():
            rss = _tree_rss_kb(self.pid)
            if rss:
                with self._lock:
                    if rss > self.peak_kb:
                        self.peak_kb = rss
                    if self._phase is not None and rss > self._phase_peak.get(self._phase, 0):
                        self._phase_peak[self._phase] = rss
            self._stop_ev.wait(self.interval)

    def start_phase(self, name):
        with self._lock:
            self._phase = name
            self._phase_peak.setdefault(name, _tree_rss_kb(self.pid))

    def end_phase(self):
        with self._lock:
            self._phase = None

    def phase_peak_mb(self, name):
        with self._lock:
            return self._phase_peak.get(name, 0) / 1024.0

    def stop(self):
        self._stop_ev.set()


class Server:
    def __init__(self, datadir, port=None, binary=None, keep_datadir=False, ready_timeout=120):
        self.datadir = os.path.abspath(datadir)
        self.port = port or find_free_port()
        self.binary = binary or DEFAULT_BINARY
        self.keep_datadir = keep_datadir
        self.ready_timeout = ready_timeout
        self.proc = None
        self.attached_pid = None
        self.sampler = None
        self.log_path = self.datadir + ".log"

    @classmethod
    def attach(cls, pid, port, datadir, keep_datadir=True):
        """Attach to an already-running serened (started by build_index.py)."""
        s = cls(datadir, port=port, keep_datadir=keep_datadir)
        s.attached_pid = pid
        s.sampler = RssSampler(pid)
        s.sampler.start()
        return s

    def _pid(self):
        if self.proc is not None:
            return self.proc.pid
        return self.attached_pid

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _spawn(self):
        logf = open(self.log_path, "a" if os.path.exists(self.log_path) else "w")
        self._logf = logf
        self.proc = subprocess.Popen(
            [self.binary, self.datadir, f"--listen=postgres://0.0.0.0:{self.port}"],
            stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self.sampler = RssSampler(self.proc.pid)
        self.sampler.start()
        self._wait_ready()

    def start(self):
        if not os.path.exists(self.binary):
            raise FileNotFoundError(
                f"serened binary not found at {self.binary}; build it with "
                f"`ninja -C build_bench serened`")
        if os.path.isdir(self.datadir):
            shutil.rmtree(self.datadir)
        os.makedirs(os.path.dirname(self.datadir) or ".", exist_ok=True)
        self._spawn()
        return self

    def restart(self):
        """Stop the process but keep the datadir, then relaunch against it.

        InvertedIndexStorage's index-open path sweeps unreferenced segment
        files (IndexWriter::Make), which is otherwise only done by the
        background refresh loop -- VACUUM (COMPACT_TABLE) itself never
        reclaims the segments it supersedes. Callers that disable the
        background loop for clean timing (compaction_interval=0,
        refresh_interval=0) need this restart to get an accurate on-disk
        size after compacting.
        """
        self._stop_process()
        self._spawn()
        return self

    def _wait_ready(self):
        deadline = time.time() + self.ready_timeout
        last = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"serened exited early (code {self.proc.returncode}); see {self.log_path}")
            try:
                c = psycopg.connect(**sdb.conn_kwargs(self.port), connect_timeout=3)
                c.close()
                return
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(0.2)
        raise TimeoutError(f"serened not ready after {self.ready_timeout}s: {last}")

    def connect(self, **kw):
        return sdb.connect(self.port, **kw)

    def datadir_bytes(self):
        return dir_size_bytes(self.datadir)

    def peak_rss_mb(self):
        return self.sampler.peak_kb / 1024.0 if self.sampler else 0.0

    def hwm_mb(self):
        pid = self._pid()
        kb = _read_status_kb(pid, "VmHWM") if pid else None
        return (kb or 0) / 1024.0

    def _stop_process(self):
        if self.sampler:
            self.sampler.stop()
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        elif self.attached_pid is not None:
            try:
                os.kill(self.attached_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if getattr(self, "_logf", None):
            self._logf.close()

    def stop(self):
        self._stop_process()
        if not self.keep_datadir and os.path.isdir(self.datadir):
            shutil.rmtree(self.datadir, ignore_errors=True)

    def detach(self):
        """Stop sampling but leave serened running (for the build/query split)."""
        if self.sampler:
            self.sampler.stop()
        if getattr(self, "_logf", None):
            self._logf.close()
