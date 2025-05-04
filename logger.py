import sys
from logging import INFO, DEBUG, WARNING

class Tee(object):
    def __init__(self, name, mode='w', lvl=INFO, hijack=False):
        self.name = name
        self.file = open(f'{name}.log', mode)
        self.stdout = sys.stdout
        self.lvl = lvl
        self.hijacked = hijack
        if hijack:
            sys.stdout = self
    def __del__(self):
        if self.hijacked:
            sys.stdout = self.stdout
        self.file.close()
    def write(self, data, lvl=INFO):
        if self.lvl > lvl: return
        data = f'[{lvl}] {data}\n'
        self.file.write(data)
        self.stdout.write(data)
    def info(self, data):
        self.write(data, INFO)
    def debug(self, data):
        self.write(data, DEBUG)
    def warn(self, data):
        self.write(data, WARNING)
    def flush(self):
        self.file.flush()