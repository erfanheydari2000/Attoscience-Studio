from PyQt5.QtNetwork import QLocalServer, QLocalSocket

class SingleInstance:
    def __init__(self, key):
        self.key = key
        self.server = None

    def is_running(self):
        socket = QLocalSocket()
        socket.connectToServer(self.key)
        is_running = socket.waitForConnected(100)
        socket.abort()
        return is_running

    def start(self):
        # Cleanup in case of leftover socket
        self.server = QLocalServer()
        if self.server.isListening():
            self.server.close()
        QLocalServer.removeServer(self.key)
        self.server.listen(self.key)
