import keras.callbacks
import time
import os
from os.path import exists
from os.path import join


class TimeHistory(keras.callbacks.Callback):
    def __init__(self, saveLogDir='./'):
        super().__init__()
        self.time = 0
        self.saveLogDir = join(saveLogDir, "timeRec.txt")
        self.prepareLogFiles()

    def prepareLogFiles(self):
        if exists(self.saveLogDir):
            os.remove(self.saveLogDir)
        logFile = open(self.saveLogDir, 'w')
        logFile.close()

    def on_train_begin(self, logs={}):
        self.train_time_start = time.time()

    def on_train_end(self, logs={}):
        self.time = time.time() - self.train_time_start
        logFile = open(self.saveLogDir, 'w')
        logFile.write(str(self.time))
        logFile.close()
