"""Single-instance guard.

A tray app that is already running looks exactly like an app that failed to
launch: you double-click, a second copy starts somewhere behind the first,
and nothing on screen changes. This was reported as "the shortcut doesn't
work" and cost a long hunt before the obvious explanation surfaced.

So the second launch does not start a second copy. It hands the request to
the instance that is already running -- which shows its window and says hello
-- and exits.

Built on QLocalServer/QLocalSocket (a named pipe on Windows) rather than a
lock file, because a lock file left behind by a crash blocks every future
launch, which trades one silent failure for a worse one.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger("instance")

SERVER_NAME = "ai-simultaneous-interpretation"
CONNECT_TIMEOUT_MS = 500
PING = b"show\n"


class SingleInstance(QObject):
    """Owns the local server. `activated` fires when another launch arrives."""

    activated = Signal()

    def __init__(self, name: str = SERVER_NAME, parent=None) -> None:
        super().__init__(parent)
        self._name = name
        self._server: QLocalServer | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def ping_existing(name: str = SERVER_NAME) -> bool:
        """True if another instance answered (so this one should exit)."""
        sock = QLocalSocket()
        sock.connectToServer(name)
        if not sock.waitForConnected(CONNECT_TIMEOUT_MS):
            return False
        sock.write(PING)
        sock.flush()
        sock.waitForBytesWritten(CONNECT_TIMEOUT_MS)
        sock.disconnectFromServer()
        return True

    def listen(self) -> bool:
        """Become the owner. Returns False if the name could not be claimed."""
        server = QLocalServer(self)
        # A hard kill leaves the pipe behind on some platforms; without this
        # the next launch can never claim the name again.
        QLocalServer.removeServer(self._name)
        if not server.listen(self._name):
            log.warning("cannot claim instance name %s: %s",
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
