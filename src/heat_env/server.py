import socket
import json
class Client(object):

    def __init__(self):
        self.HOST = 'localhost'
        self.PORT = 5555
        self.ADDR = (self.HOST, self.PORT)
        self.BUFSIZE = 4096
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send_data(self,data):
        #add exception catching and try finally
        self.client.connect(self.ADDR)
        self.client.send(json.dumps(data) )
        self.client.close()

class Server(object):

    def __init__(self):
        self.HOST = 'localhost'
        self.PORT = 5555
        self.ADDR = (self.HOST, self.PORT)
        self.BUFSIZE = 4096
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.server.bind(self.ADDR)
        self.server.listen(5)

    def listening(self):
        print('Start listening from server')
        while True:
            conn, addr = self.server.accept()
            print('client connected ... ', addr)

            while True:
                data = conn.recv(self.BUFSIZE)
                if not data: break
                print('caught data...')
                print(data)

            print('closing connection')
            conn.close()
            print('client disconnected')

if __name__ == '__main__':
    serv = Server()
    serv.listening()

