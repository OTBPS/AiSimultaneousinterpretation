"""Single-instance guard.

A tray app that is already running looks exactly like an app that failed to
launch: you double-click, a second copy starts somewhere behind the first,
and nothing on screen changes. This was reported as "the shortcut doesn't
work" and cost a long hunt before the obvious explanation surfaced.

So the second launch does not start a second copy. It hands the request to
the instance that is already running -- which shows its window and says hello
-- and exits.

Exclusivity comes from a named mutex, not from the socket. QLocalServer looked
like it would serve both jobs, but on Windows a named pipe accepts additional
server instances under the same name -- listen() succeeded for two processes
launched in the same second and both ran to completion, which is precisely
what this module exists to stop. CreateMutexW is atomic and reports
ERROR_ALREADY_EXISTS, so it decides who owns the session; the socket is left
to do the one thing it is good at, carrying "please show yourself" to the
owner.

A mutex also beats a lock file: the kernel releases it when the process dies,
so a crash cannot leave something behind that blocks every future launch.
"""
from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger("instance")

SERVER_NAME = "ai-simultaneous-interpretation"
MUTEX_NAME = r"Local\ai-simultaneous-interpretation"
CONNECT_TIMEOUT_MS = 500
PING = b"show\n"
_ERROR_ALREADY_EXISTS = 183


def claim_mutex(name: str = MUTEX_NAME):
    """(handle, we_are_first). The handle must outlive the process.

    CreateMutexW is atomic and reports ERROR_ALREADY_EXISTS, which is what
    makes it a usable lock. QLocalServer.listen() is not: on Windows a named
    pipe accepts additional server instances under the same name, so two
    launches in the same second both succeeded and both ran.

    Non-Windows returns "first" so the guard degrades to a no-op rather than
    blocking startup.
    """
    if sys.platform != "win32":
        return None, True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [wintypes.LPCVOID, wintypes.BOOL,
                                      wintypes.LPCWSTR]
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return None, True                    # cannot tell; never block
    return handle, kernel32.GetLastError() != _ERROR_ALREADY_EXISTS


def notify_existing(name: str = SERVER_NAME) -> bool:
    """Tell the running instance to show itself. True if it answered."""
    sock = QLocalSocket()
    sock.connectToServer(name)
    if not sock.waitForConnected(CONNECT_TIMEOUT_MS):
        return False
    sock.write(PING)
    sock.flush()
    sock.waitForBytesWritten(CONNECT_TIMEOUT_MS)
    sock.disconnectFromServer()
    return True


class SingleInstance(QObject):
    """Owns the local server. `activated` fires when another launch arrives."""

    activated = Signal()

    def __init__(self, name: str = SERVER_NAME, parent=None) -> None:
        super().__init__(parent)
        self._name = name
        self._server: QLocalServer | None = None
        self._mutex = None

    # ------------------------------------------------------------------
    @staticmethod
    def ping_existing(name: str = SERVER_NAME) -> bool:
        """True if another instance answered (so this one should exit)."""
        return notify_existing(name)

    def claim(self) -> bool:
        """Atomically become the one instance. False means someone else is.

        The mutex, not the socket, is the decision: two same-second launches
        both got listen() to succeed on Windows and both ran.
        """
        self._mutex, first = claim_mutex()
        return first

    def listen(self) -> bool:
        """Start serving 'show yourself' requests. Call only after claim()."""
        server = QLocalServer(self)
        if not server.listen(self._name):
            # A hard kill can leave the pipe behind; we already hold the mutex,
            # so nobody else is listening and clearing it is safe.
            QLocalServer.removeServer(self._name)
            if not server.listen(self._name):
                log.warning("cannot serve on %s: %s",
                            self._name, server.errorString())
                return False
        server.newConnection.connect(self._on_connection)
        self._server = server
        return True

    def _on_connection(self) -> None:
        sock = self._server.nextPendingConnection() if self._server else None
        if sock is None:
            return
        sock.readyRead.connect(lambda: sock.readAll())
        sock.disconnected.connect(sock.deleteLater)
        log.info("another launch arrived; showing the existing window")
        self.activated.emit()

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            QLocalServer.removeServer(self._name)
            self._server = None
        if self._mutex is not None:
            try:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._mutex)
            except Exception:                              # noqa: BLE001
                pass
            self._mutex = None
