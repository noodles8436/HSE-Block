from keras import layers, models
import SECallback
import SELayer
import TimeRecorder
import metrics
import loadDatasets
from keras.callbacks import CSVLogger

# !! Config : Please Setting !!
# 무지성 증가는 False, 알아서 찾아가기는 True 값으로 변경!

ClassNum = 6
MAX_EPOCHS = 600
batch_size = 16
split_rate = 0.2
img_channel = 3
img_size = 224


# Load >> VEGETAGLE << Dataset Using TFDS API
train_folder = 'D:\\Research\\Datasets\\Intel Image Classification\\archive\\seg_train\\seg_train'
val_folder = 'D:\\Research\\Datasets\\Intel Image Classification\\archive\\seg_test\\seg_test'
train_ds = loadDatasets.getTFDS(data_dir=train_folder, img_size=img_size,
                                batch_size=batch_size)
val_ds = loadDatasets.getTFDS(data_dir=val_folder, img_size=img_size,
                              batch_size=batch_size)

'''
# Load >> FRUIT << Dataset Using TFDS API
dataset_dir = 'E:\\catVsDog\\PetImages'

train_ds, val_ds = loadDatasets.getTFDS_split(data_dir=dataset_dir, img_size=img_size,
                                              batch_size=batch_size, split_rate=0.2)
                                              
'''
# Create HSE Blocks
my_HSE1 = SELayer.SEBlock(ratio=2, init_filter=32, max_filter=256, kernel_size=(3, 3),
                          layerName="SEBLock1", bottomTrain=False, skip=False, batch_norm=False)
my_HSE2 = SELayer.SEBlock(ratio=2, init_filter=32, max_filter=256, kernel_size=(3, 3),
                          layerName="SEBLock2", bottomTrain=False, skip=False, batch_norm=False)
my_HSE3 = SELayer.SEBlock(ratio=2, init_filter=32, max_filter=256, kernel_size=(3, 3),
                          layerName="SEBLock3", bottomTrain=False, skip=False, batch_norm=False)

# Create HSE Callbacks & csvLogger
listOfHSE = [my_HSE1, my_HSE2, my_HSE3]
HSECallback = SECallback.SECallback(listOfHSE, Activation_Epoch=3)
csv_logger = CSVLogger('HSE.log', append=True, separator=':')
timeRec = TimeRecorder.TimeHistory()

# Create Models Using HSE Blocks
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(img_size, img_size, img_channel)))
model.add(my_HSE1)
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(my_HSE2)
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(my_HSE3)
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(ClassNum, activation='softmax'))

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', metrics.f1_m, metrics.recall_m, metrics.precision_m], run_eagerly=True)

model.summary()

model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, callbacks=[csv_logger, HSECallback, timeRec])