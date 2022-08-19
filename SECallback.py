import os
from keras import callbacks
from os.path import exists
from os.path import join


def weightsToSTR(weights):
    result = ""
    listOfWeights = weights
    for w in listOfWeights:
        result += str(w)
    return result


class SECallback(callbacks.Callback):

    def __init__(self, listOfHSE, Activation_Epoch, saveLogDir="./", testMode=False):
        self.listOfHSE = listOfHSE
        self.Activation_Epoch = Activation_Epoch
        self.saveLogDir = saveLogDir
        self.prepareLogFiles()
        self.SEWeightsRec = dict()
        self.testMode = testMode

    def prepareLogFiles(self):
        for hse in self.listOfHSE:
            path = self.getPath(hse.getHSEName())
            if exists(path):
                os.remove(path)
            logFile = open(path, 'w')
            logFile.close()

    def on_train_batch_end(self, batch, logs=None):
        for hse in self.listOfHSE:
            hse.refreshWeights()

    def on_epoch_end(self, epoch, logs=None):
        print('\n[ HSE BLOCK CallBack ] Start to Check HSE BLOCK...\n')

        for hse in self.listOfHSE:
            if (epoch + 1) % self.Activation_Epoch == 0:
                weights_mean = hse.Activate()
            else:
                _, weights_mean = hse.check_SEWeights()

            weights_mean = weightsToSTR(weights_mean)

            path = self.getPath(hse.getHSEName())
            logFile = open(path, 'a')
            logFile.writelines(weights_mean + "\n")
            logFile.close()

            hse.clearWeightRec()

        print('[ HSE BLOCK CallBack ] Done!\n')

    def getPath(self, HSEName):
        HSEName = HSEName + ".txt"
        path = join(self.saveLogDir, HSEName)
        return path
