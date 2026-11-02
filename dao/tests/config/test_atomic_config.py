"""Real file/process boundaries for config publication and transactions.

The pipes are barriers, not timing guesses: each assertion observes a writer
paused at a real write/replace boundary. Child processes never run pytest.
"""

import builtins
import errno
import json
import multiprocessing
import os
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from dao.prog.config import loader


def _child_entry(connection, target, args):
    try:
        target(connection, *args)
        connection.send(("done",))
    except BaseException as error:
        # Do not send document/secret contents through diagnostics.
        connection.send(("error", type(error).__name__))
        raise SystemExit(1) from None
    finally:
        connection.close()


@contextmanager
def _managed_process(target, *args):
    """Own every child, including termination when a boundary assertion fails."""
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=_child_entry, args=(child, target, args))
    process.start()
    child.close()
    try:
        yield parent
    finally:
        parent.close()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


def _message(connection, expected):
    assert connection.poll(15), f"Child did not reach {expected} boundary"
    message = connection.recv()
    assert message[0] == expected, message
    print("B8c boundary:", *message)
    return message[1:]


class _WriteProbe:
    def __init__(self, handle, write):
        self.handle = handle
        self._write = write

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def write(self, text):
        return self._write(self.handle, text)


def _intercept_text_writes(monkeypatch, write):
    """Instrument real streams, for both the old open and exclusive fd open."""
    original_open = getattr(loader, "open", builtins.open)
    original_fdopen = os.fdopen

    def wrap_open(file, mode="r", *args, **kwargs):
        handle = original_open(file, mode, *args, **kwargs)
        return _WriteProbe(handle, write) if mode == "w" else handle

    def wrap_fdopen(fd, mode="r", *args, **kwargs):
        handle = original_fdopen(fd, mode, *args, **kwargs)
        return _WriteProbe(handle, write) if mode == "w" else handle

    monkeypatch.setattr(loader, "open", wrap_open, raising=False)
    monkeypatch.setattr(os, "fdopen", wrap_fdopen)


def _pause_replace(connection):
    original_replace = os.replace

    def replace(source, target):
        connection.send(("before_replace", os.stat(source).st_ino))
        assert connection.recv() == "publish"
        original_replace(source, target)
        connection.send(("after_replace", os.stat(target).st_ino))
        assert connection.recv() == "unlock"

    loader.os.replace = replace


def _paused_config_writer(connection, path, operation):
    _pause_replace(connection)
    if operation == "migration":
        loader.ConfigurationLoader(path).load_and_validate()
    else:
        assert operation == "mode"
        loader.set_fast_control_mode(path, "active")


def _probe_transaction(connection, path):
    """Prove actual kernel contention, and that no config read preceded it."""
    original_flock = loader.fcntl.flock
    original_load = json.load
    original_read_bytes = Path.read_bytes

    def flock(fd, operation):
        if operation == loader.fcntl.LOCK_EX:
            try:
                original_flock(fd, operation | loader.fcntl.LOCK_NB)
            except BlockingIOError:
                connection.send(("blocked", os.fstat(fd).st_ino))
                original_flock(fd, operation)
            else:
                connection.send(("unblocked", os.fstat(fd).st_ino))
        else:
            original_flock(fd, operation)

    def load(handle, *args, **kwargs):
        if Path(handle.name) == path:
            connection.send(("read",))
        return original_load(handle, *args, **kwargs)

    def read_bytes(self):
        if self == path:
            connection.send(("read",))
        return original_read_bytes(self)

    loader.fcntl.flock = flock
    loader.json.load = load
    Path.read_bytes = read_bytes


def _waiting_config_writer(connection, path, operation):
    _probe_transaction(connection, path)
    if operation == "migration":
        loader.ConfigurationLoader(path).load_and_validate()
    else:
        assert operation == "mode"
        loader.set_fast_control_mode(path, "active")


def _atomic_writer(connection, path, text, partial):
    if partial:
        def write(handle, content):
            half = len(content) // 2
            handle.write(content[:half])
            handle.flush()
            connection.send(("partial_write", os.fstat(handle.fileno()).st_ino))
            assert connection.recv() == "finish"
            handle.write(content[half:])
            return len(content)

        with pytest.MonkeyPatch.context() as monkeypatch:
            _intercept_text_writes(monkeypatch, write)
            loader.atomic_write_text(path, text)
    else:
        original_replace = os.replace

        def replace(source, target):
            connection.send(("before_replace", os.stat(source).st_ino))
            assert connection.recv() == "publish"
            original_replace(source, target)

        loader.os.replace = replace
        loader.atomic_write_text(path, text)


def test_two_config_writers_never_publish_partial_json(tmp_path):
    """A publishing writer must not rename another writer's half-written temp."""
    path = tmp_path / "options.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    first = json.dumps({"writer": "first", "padding": "a" * 16384}) + "\n"
    second = json.dumps({"writer": "second", "padding": "b" * 32768}) + "\n"

    with _managed_process(_atomic_writer, path, first, False) as writer_a:
        (inode_a,) = _message(writer_a, "before_replace")
        with _managed_process(_atomic_writer, path, second, True) as writer_b:
            (inode_b,) = _message(writer_b, "partial_write")
            writer_a.send("publish")
            _message(writer_a, "done")
            # B is still paused with only half its stream flushed to disk.
            published = path.read_text(encoding="utf-8")
            assert published == first, "A must publish its own complete document"
            assert json.loads(published)["writer"] == "first"
            assert inode_a != inode_b
            writer_b.send("finish")
            _message(writer_b, "done")

    assert path.read_text(encoding="utf-8") == second
    assert set(tmp_path.iterdir()) == {path}


@pytest.mark.parametrize("stage", ["write", "fsync", "replace"])
def test_atomic_write_cleans_own_temp_on_failure(tmp_path, monkeypatch, stage):
    """Every pre-publication failure leaves old bytes and other temps alone."""
    path = tmp_path / "options.json"
    path.write_bytes(b'{"old": true}\n')
    other_temp = tmp_path / ".options.json.other-writer.tmp"
    other_temp.write_bytes(b"owned by a different writer")
    before = {file: file.read_bytes() for file in tmp_path.iterdir()}
    primary = OSError(f"Synthetic {stage} failure")

    if stage == "write":
        def write(handle, text):
            handle.write(text[:4])
            handle.flush()
            raise primary

        _intercept_text_writes(monkeypatch, write)
    elif stage == "fsync":
        original_fsync = os.fsync

        def fsync(fd):
            original_fsync(fd)
            raise primary

        monkeypatch.setattr(os, "fsync", fsync)
    else:
        def replace(source, target):
            raise primary

        monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(OSError) as caught:
        loader.atomic_write_text(path, '{"new": true}\n')

    assert caught.value is primary
    assert {file: file.read_bytes() for file in tmp_path.iterdir()} == before


@pytest.mark.parametrize("stage", ["write", "flush", "fsync", "close"])
def test_atomic_write_stream_failure_keeps_original_error_and_old_file(
    tmp_path, monkeypatch, stage
):
    """A second close error must not mask a primary error or permit publication."""
    path = tmp_path / "options.json"
    path.write_bytes(b'{"old": true}\n')
    other_temp = tmp_path / ".options.json.other-writer.tmp"
    other_temp.write_bytes(b"owned by a different writer")
    before = {file: file.read_bytes() for file in tmp_path.iterdir()}
    primary = OSError(f"Synthetic {stage} failure")
    close_error = primary if stage == "close" else OSError("Separate synthetic close failure")
    original_fdopen = os.fdopen
    original_fsync = os.fsync
    handles = []

    class FailingStream:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def write(self, text):
            if stage == "write":
                self.handle.write(text[:4])
                self.handle.flush()
                raise primary
            return self.handle.write(text)

        def flush(self):
            self.handle.flush()
            if stage == "flush":
                raise primary

        def close(self):
            self.handle.close()
            raise close_error

    def fdopen(*args, **kwargs):
        handle = original_fdopen(*args, **kwargs)
        handles.append(handle)
        return FailingStream(handle)

    def fsync(fd):
        original_fsync(fd)
        if stage == "fsync":
            raise primary

    monkeypatch.setattr(os, "fdopen", fdopen)
    monkeypatch.setattr(os, "fsync", fsync)
    try:
        with pytest.raises(OSError) as caught:
            loader.atomic_write_text(path, '{"new": true}\n')

        assert path.read_bytes() == b'{"old": true}\n'
        assert {file: file.read_bytes() for file in tmp_path.iterdir()} == before
        assert handles and all(handle.closed for handle in handles)
        assert caught.value is primary
    finally:
        for handle in handles:
            handle.close()


def test_atomic_write_cleanup_cannot_mask_primary_error(tmp_path, monkeypatch):
    path = tmp_path / "options.json"
    path.write_bytes(b'{"old": true}\n')
    primary = OSError("Synthetic publication failure")
    cleanup_attempts = []

    def replace(source, target):
        raise primary

    def unlink(file, *args, **kwargs):
        cleanup_attempts.append(Path(file))
        raise PermissionError("Synthetic cleanup failure")

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(OSError) as caught:
        loader.atomic_write_text(path, '{"new": true}\n')

    assert caught.value is primary
    assert path.read_bytes() == b'{"old": true}\n'
    assert len(cleanup_attempts) == 1
    assert cleanup_attempts[0].parent == tmp_path
    assert cleanup_attempts[0] != path


def test_atomic_write_closes_temp_descriptor_when_stream_open_fails(tmp_path, monkeypatch):
    path = tmp_path / "options.json"
    path.write_bytes(b"{}\n")
    primary = OSError("Synthetic stream creation failure")
    descriptors = []

    def fdopen(fd, *args, **kwargs):
        descriptors.append(fd)
        raise primary

    monkeypatch.setattr(os, "fdopen", fdopen)
    with pytest.raises(OSError) as caught:
        loader.atomic_write_text(path, '{"new": true}\n')

    assert caught.value is primary
    assert path.read_bytes() == b"{}\n"
    assert set(tmp_path.iterdir()) == {path}
    try:
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[0])
        assert closed.value.errno == errno.EBADF
    finally:
        # A RED run must not leak the descriptor it just exposed.
        try:
            os.close(descriptors[0])
        except OSError:
            pass


def test_atomic_secret_temp_is_exclusive_private_and_flushed(tmp_path, monkeypatch):
    """Neither an existing temp nor a permissive umask may expose secret bytes."""
    path = tmp_path / "secrets.json"
    path.write_bytes(b"{}\n")
    occupied = tmp_path / ".secrets.json.tmp"
    occupied.write_bytes(b"another writer")
    text = '{"synthetic": "sandbox-only"}\r\n'
    original_fsync = os.fsync
    original_replace = os.replace
    synced = []

    def fsync(fd):
        temp = next(file for file in tmp_path.iterdir() if file.stat().st_ino == os.fstat(fd).st_ino)
        complete = temp.read_bytes() == text.encode("utf-8")
        assert complete, "The complete text must be flushed before fsync"
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
        original_fsync(fd)
        synced.append(os.fstat(fd).st_ino)

    def replace(source, target):
        source = Path(source)
        assert source.parent == path.parent
        assert source != occupied
        assert source.stat().st_ino in synced
        assert path.read_bytes() == b"{}\n"
        original_replace(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    previous_umask = os.umask(0)
    try:
        loader.atomic_write_text(path, text)
    finally:
        os.umask(previous_umask)

    exact = path.read_bytes() == text.encode("utf-8")
    assert exact, "Secret publication must preserve the submitted bytes"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert occupied.read_bytes() == b"another writer"
    assert set(tmp_path.iterdir()) == {path, occupied}


def test_migration_and_mode_change_share_stable_transaction_lock(tmp_path):
    """Waiters before/after replace contend on one inode, before any read."""
    path = tmp_path / "options.json"
    original = {"//note": "preserved", "fast control": {"mode": "off"}}
    path.write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "secrets.json").write_text("{}", encoding="utf-8")

    with _managed_process(_paused_config_writer, path, "migration") as migration:
        _message(migration, "before_replace")
        with _managed_process(_waiting_config_writer, path, "mode") as mode:
            (lock_inode,) = _message(mode, "blocked")
            migration.send("publish")
            (published_inode,) = _message(migration, "after_replace")
            with _managed_process(_waiting_config_writer, path, "migration") as late:
                (late_lock_inode,) = _message(late, "blocked")
                assert late_lock_inode == lock_inode != published_inode
                migration.send("unlock")
                _message(migration, "done")
                _message(mode, "read")
                _message(mode, "done")
                _message(late, "read")
                _message(late, "done")

    saved = json.loads(path.read_bytes())
    assert saved["fast control"]["mode"] == "active"
    assert saved["//note"] == "preserved"
    assert saved["config_version"] == loader.CURRENT_VERSION
    assert json.loads((tmp_path / "options_unversioned.json").read_bytes()) == original
    locks = [file for file in tmp_path.iterdir() if file.stat().st_ino == lock_inode]
    assert len(locks) == 1, "The stable sidecar must survive the whole transaction"
    loader.set_fast_control_mode(path, "off")
    assert locks[0].stat().st_ino == lock_inode
