import os
import numpy as np
import matplotlib.pyplot as plt


def load_weights(log_dir):
    file = open(log_dir, 'r')
    lines = file.readlines()
    logs = list()
    epoch_log = None

    for line in lines:
        line = str(line)
        if line.find("[[") != -1:
            if epoch_log is not None:
                logs.append(epoch_log)
            epoch_log = list()
            line = line.replace("[[", "")

        line = line.replace(']]', "")

        datas = line.split(" ")
        for value in datas:
            value = value.strip()
            if value != '':
                epoch_log.append(float(value))

    logs.append(epoch_log)

    return logs


def load_trainLog(log_dir):
    file = open(log_dir, 'r')
    lines = file.readlines()
    logs = list()

    line = str(lines[0]).strip().split(":")
    logs.append(line[1:])

    lines = lines[1:]

    for line in lines:
        line = str(line).strip().split(":")
        values = list()
        for value in line[1:]:
            values.append(float(value))
        logs.append(values)

    return logs


def drawEachWeights(data, label='Each Weight'):
    epoch = np.arange(len(data))
    weights = []
    for i in range(len(data)):
        if len(weights) < len(data[i]):
            for i in range(len(data[i]) - len(weights)):
                weights.append(list())
        for wi in range(len(data[i])):
            weights[wi].append(data[i][wi])

    slot = np.ceil(np.sqrt(len(data[-1])))
    for i in range(len(weights)):
        plt.subplot(slot, slot, i + 1)
        sub_epoch = np.arange(epoch[-1] - len(weights[i]), epoch[-1])
        plt.plot(sub_epoch, weights[i])
        plt.xlim(0, epoch[-1])
        plt.ylim(0, 1)

    plt.show()


def drawAverage(data, label='Weights Avg'):
    epoch = np.arange(len(data))
    average = []
    for i in range(len(data)):
        average.append(np.mean(data[i]))
    plt.ylim(0, 1)
    plt.plot(epoch, average, label=label)

    return average


def drawStd(data, label='Weight Std'):
    epoch = np.arange(len(data))
    std = []
    for i in range(len(data)):
        std.append(np.std(data[i]))
    plt.ylim(0, 1)
    plt.plot(epoch, std, label=label)


def drawLastHist(data, title='SEBlock', bins=20):
    plt.hist(data[-1], bins=bins)
    plt.title(title)
    plt.show()


def drawWeightRate(data, label='High W Rate', rateThres=0.9, color='#f542bc'):
    epoch = np.arange(len(data))
    rate = []
    for i in range(len(data)):
        sum = 0
        for wi in range(len(data[i])):
            if data[i][wi] >= rateThres:
                sum += data[i][wi]
        _epoch_rate = sum / len(data[i])
        rate.append(_epoch_rate)
    plt.ylim(0, 1)
    plt.plot(epoch, rate, label=label, color=color)

    return rate


def drawTrainLog(data, axis='accuracy', color='forestgreen', label='accuracy'):
    epoch = np.arange(len(data) - 1)
    train_acc = list()

    try:
        index = data[0].index(axis)
    except ValueError as e:
        print("존재하지 않는 Axis 축 : ", axis)
        return

    data = data[1:]

    for i in range(len(data)):
        train_acc.append(data[i][index])

    plt.ylim(0, 1)
    plt.plot(epoch, train_acc, color=color, label=label)

'''
trainLog = load_trainLog("./Final/fruit_batch16/fruit_false/HSE.log")
weights_datas = load_weights("./Final/fruit_batch16/fruit_false/SEBLock1.txt")

trainLog = load_trainLog("./Final/fruit_batch16/fruit_true/HSE.log")
weights_datas = load_weights("./Final/fruit_batch16/fruit_true/SEBLock1.txt")

trainLog = load_trainLog("./Final/vegetable_batch16/vegetable_false/HSE.log")
weights_datas = load_weights("./Final/vegetable_batch16/vegetable_false/SEBLock1.txt")

trainLog = load_trainLog("./Final/vegetable_batch16/vegetable_true/HSE.log")
weights_datas = load_weights("./Final/vegetable_batch16/vegetable_true/SEBLock1.txt")
'''

seBlockRange = 3
folder = "./"

for i in range(seBlockRange):
    seBlockNum = i + 1
    trainLog = load_trainLog(str(folder) + "/HSE.log")
    weights_datas = load_weights(str(folder) + "/SEBLock" + str(seBlockNum) + ".txt")

    drawTrainLog(trainLog, axis='accuracy', label='train_acc')
    drawTrainLog(trainLog, axis='val_accuracy', color='#e35f62', label='val_acc')

    drawWeightRate(weights_datas, label='0.9 Ratio', rateThres=0.9, color='#8B0000')

    plt.axhline(0.2, 0, 1, color='#008275', linestyle='--', linewidth=2)
    plt.title("Intel : SEBLOCK " + str(seBlockNum), fontsize=16)
    plt.legend(loc='center right', ncol=1, fontsize=18)
    plt.tight_layout()
    plt.savefig('seblock' + str(seBlockNum), bbox_inches='tight')
    plt.show()

'''
weights_datas = load_weights("./Final/fruit_batch16/fruit_false/SEBLock1.txt")
print(len(weights_datas[-1]))
weights_datas = load_weights("./Final/fruit_batch16/fruit_true/SEBLock1.txt")
print(len(weights_datas[-1]))
weights_datas = load_weights("./Final/vegetable_batch16/vegetable_false/SEBLock1.txt")
print(len(weights_datas[-1]))
weights_datas = load_weights("./Final/vegetable_batch16/vegetable_true/SEBLock1.txt")
print(len(weights_datas[-1]))

print('\n\n')

f = open('./FInal/fruit_batch16/fruit_false/timeRec.txt', 'r')
fruit_false_time = float(f.readline())
f = open('./FInal/fruit_batch16/fruit_true/timeRec.txt', 'r')
fruit_true_time = float(f.readline())
f = open('./FInal/vegetable_batch16/vegetable_false/timeRec.txt', 'r')
vegetable_false_time = float(f.readline())
f = open('./FInal/vegetable_batch16/vegetable_true/timeRec.txt', 'r')
vegetable_true_time = float(f.readline())

print(fruit_false_time)
print(fruit_true_time)
print(vegetable_false_time)
print(vegetable_true_time)

print('\n\n')

f_time = (fruit_false_time - fruit_true_time) / 60
t_time = (vegetable_false_time - vegetable_true_time) / 60

print(f_time, t_time)

print('\n\n')

f_time_per = (abs(fruit_false_time-fruit_true_time) / ((fruit_false_time-fruit_true_time)/2)) * 100
v_time_per = (abs(vegetable_false_time-vegetable_true_time) / ((vegetable_false_time-vegetable_true_time)/2)) * 100

print('time per : Fruit : ', f_time_per)
print('time per : Vegetable : ', v_time_per)

print('\n\n')

trainLog = load_trainLog("./Final/Fruit_batch16/Fruit_false/HSE.log")
FF_val_acc = float(trainLog[-1][5])
FF_val_loss = float(trainLog[-1][7])
FF_val_f1 = float(trainLog[-1][6])

print('Fruit False')
print('val acc : ', FF_val_acc, 'val Loss : ', FF_val_loss, 'val f1 : ', FF_val_f1)

trainLog = load_trainLog("./Final/Fruit_batch16/Fruit_true/HSE.log")
FT_val_acc = float(trainLog[-1][5])
FT_val_loss = float(trainLog[-1][7])
FT_val_f1 = float(trainLog[-1][6])

print('Fruit True')
print('val acc : ', FT_val_acc, 'val Loss : ', FT_val_loss, 'val f1 : ', FT_val_f1)

trainLog = load_trainLog("./Final/vegetable_batch16/vegetable_false/HSE.log")
VF_val_acc = trainLog[-1][5]
VF_val_loss = trainLog[-1][7]
VF_val_f1 = trainLog[-1][6]

print('vegetable False')
print('val acc : ', VF_val_acc, 'val Loss : ', VF_val_loss, 'val f1 : ', VF_val_f1)

trainLog = load_trainLog("./Final/vegetable_batch16/vegetable_true/HSE.log")
VT_val_acc = trainLog[-1][5]
VT_val_loss = trainLog[-1][7]
VT_val_f1 = trainLog[-1][6]

print('vegetable True')
print('val acc : ', VT_val_acc, 'val Loss : ', VT_val_loss, 'val f1 : ', VT_val_f1)

print('Fruit')
print('val acc diff : ', str((FF_val_acc-FT_val_acc)*100), 'val loss diff : ', FF_val_loss-FT_val_loss,
      'val f1 diff : ', FF_val_f1- FT_val_f1)

print('\nVegetable')
print('val acc diff : ', str((VF_val_acc-VT_val_acc)*100), 'val loss diff : ', VF_val_loss-VT_val_loss,
      'val f1 diff : ', VF_val_f1-VT_val_f1)
'''