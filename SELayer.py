import keras.layers
import numpy as np
from keras.layers import Layer
from keras import initializers
import tensorflow as tf


class SEBlock(Layer):

    def __init__(self, ratio, init_filter, max_filter, kernel_size, layerName=None,
                 bottomTrain=False, skip=False, batch_norm=False, **kwargs):
        self.ratio = ratio
        self.SEWeights = None
        self.lastWeights = None
        self.batchCnt = 0
        self.init_filter = init_filter
        self.filter_num = init_filter
        self.SE_inner_node = max((self.filter_num // self.ratio), 1)
        self.max_filter = max_filter
        self.kernel_size = kernel_size
        self.layerName = layerName
        self.bottomTrain = bottomTrain
        self.skipConnection = skip
        self.batchNormalization = batch_norm
        super(SEBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_channel = input_shape[3]
        self.prepare()

    def call(self, inputs):

        identity = inputs

        x = self.TopConv(inputs)
        if self.batchNormalization:
            x = self.batchNormLayer(x)

        x = self.TopConvActivation(x)

        ex = self.GAP(x)
        ex = self.Reshape(ex)
        ex = self.SEDense_In(ex)
        ex = self.SEDense_Out(ex)
        y = self.Scaler([x, ex])

        y = self.BottomConv(y)
        if self.batchNormalization:
            y = self.batchNormLayer(y)
        y = self.BottomConvActivation(y)

        if self.skipConnection:
            if inputs.shape.as_list()[-1] != self.init_filter:
                identity = self.identityLayer(x)
            y = self.Sum([y, identity])

        self.lastWeights = ex
        self.batchCnt += 1

        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def clearWeightRec(self):
        self.SEWeights = None
        self.lastWeights = None
        self.batchCnt = 0

    def refreshWeights(self):
        if self.SEWeights is None:
            self.SEWeights = np.mean(self.lastWeights.numpy(), axis=0)
        else:
            self.SEWeights = (self.SEWeights * self.batchCnt + np.mean(self.lastWeights.numpy(), axis=0)) \
                             / (self.batchCnt + 1)
        self.batchCnt += 1

    def prepare(self):
        self.TopConv = keras.layers.Conv2D(name='SE-TopConv', filters=self.filter_num,
                                           kernel_size=self.kernel_size, padding='same',
                                           kernel_initializer='he_normal')

        self.TopConvActivation = keras.layers.ReLU()

        self.GAP = keras.layers.GlobalAveragePooling2D()

        self.Reshape = keras.layers.Reshape((1, 1, self.filter_num))

        self.SEDense_In = keras.layers.Dense(name='SE-Dense_In', units=(self.filter_num // self.ratio),
                                             activation='relu', kernel_initializer='he_normal', use_bias=False)

        self.SEDense_Out = keras.layers.Dense(name='SE-Dense_Out', units=self.filter_num,
                                              activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

        self.BottomConv = keras.layers.Conv2D(name='SE-Bottom', filters=self.init_filter,
                                              kernel_size=self.kernel_size, padding='same',
                                              kernel_initializer='he_normal', trainable=self.bottomTrain)

        self.BottomConvActivation = keras.layers.ReLU()

        self.batchNormLayer = keras.layers.BatchNormalization()

        self.Scaler = keras.layers.Multiply()

        self.identityLayer = keras.layers.Conv2D(filters=self.init_filter, kernel_size=1, padding='same')

        self.Sum = keras.layers.Add()

    def Activate(self):
        isIncrease, weights_mean = self.check_SEWeights()
        if isIncrease:
            self.increase()
        return weights_mean

    def check_SEWeights(self):
        weights = self.SEWeights
        print(self.layerName,' : ', weights, '\n')
        sum = 0
        for i in range(len(weights[0][0])):
            if weights[0][0][i] >= 0.9:
                sum += weights[0][0][i]
        rate = sum / len(weights[0][0])

        if rate > 0.2 and self.filter_num < self.max_filter:
            return True, weights
        return False, weights

    def increase(self):
        print('========== [Increase Before] ============\n')
        print('TopConv >> ', self.TopConv.get_weights()[0].shape)
        # print('TopConv \n', self.TopConv.get_weights()[0])
        print('SEDense_In >> ', self.SEDense_In.get_weights()[0].shape)
        # print('SEDense_In >> ', self.SEDense_In.get_weights()[0])
        print('SEDense_Out >> ', self.SEDense_Out.get_weights()[0].shape)
        # print('SEDense_Out >> ', self.SEDense_Out.get_weights()[0])
        print('BottomConv >> ', self.BottomConv.get_weights()[0].shape)
        # print('BottomConv >> ', self.BottomConv.get_weights()[0])

        # Prepare New Weights
        topInit = initializers.initializers_v2.HeNormal()
        self.topConvSample = self.TopConv.get_weights()[0].transpose(3, 2, 0, 1)[0]
        self.topConvSample = topInit(shape=(1,) + self.topConvSample.shape).numpy()

        SEDense_In_Init = initializers.initializers_v2.HeNormal()
        self.SEDense_In_Sample = self.SEDense_In.get_weights()[0][0]
        self.SEDense_In_Sample = SEDense_In_Init(shape=(1,) + self.SEDense_In_Sample.shape)

        SEDense_Out_Init = initializers.initializers_v2.HeNormal()
        self.SEDense_Out_Sample = self.SEDense_Out.get_weights()[0].T[0]
        self.SEDense_Out_Sample = SEDense_Out_Init(shape=self.SEDense_Out_Sample.shape).numpy().reshape((1, -1))

        bottomInit = initializers.initializers_v2.HeNormal()
        self.bottomConvSample = self.BottomConv.get_weights()[0].transpose(3, 2, 0, 1)[0][0]
        self.bottomConvSample = np.array([bottomInit(shape=self.bottomConvSample.shape).numpy()])


        # Transpose Layers Weights
        topConv_Weight = self.TopConv.get_weights()[0].transpose(3, 2, 0, 1)
        topConv_Bias = self.TopConv.get_weights()[1]

        seDense1_Weight = self.SEDense_In.get_weights()[0]

        seDense2_Weight = self.SEDense_Out.get_weights()[0]

        bottomConv_Weight = self.BottomConv.get_weights()[0].transpose(3, 2, 0, 1)
        bottomConv_Bias = self.BottomConv.get_weights()[1]



        # ADD SAMPLE IN WIEGHTS
        topConv_Weight = np.concatenate((topConv_Weight, self.topConvSample), axis=0).transpose(2, 3, 1, 0)
        topConv_Bias = np.append(topConv_Bias, 0)

        seDense1_Weight = np.concatenate((seDense1_Weight, self.SEDense_In_Sample), axis=0)
        seDense2_Weight = np.row_stack([seDense2_Weight.T, self.SEDense_Out_Sample]).T

        if (self.filter_num + 1) % self.ratio == 0:
            SEDense_In_Init = initializers.initializers_v2.HeNormal()
            self.SEDense_In_Rear_Sample = seDense1_Weight.T[0]
            self.SEDense_In_Rear_Sample = SEDense_In_Init(self.SEDense_In_Rear_Sample.shape).numpy().reshape((1, -1))

            SEDense_Out_Init = initializers.initializers_v2.HeNormal()
            self.SEDense_Out_Front_Sample = seDense2_Weight[0]
            self.SEDense_Out_Front_Sample = SEDense_Out_Init(shape=(1,) + self.SEDense_Out_Front_Sample.shape)

            seDense1_Weight = np.row_stack([seDense1_Weight.T, self.SEDense_In_Rear_Sample]).T
            seDense2_Weight = np.concatenate((seDense2_Weight, self.SEDense_Out_Front_Sample), axis=0)
            self.SE_inner_node += 1


        new_w = np.array([])
        for _filter in bottomConv_Weight:
            new_filter = np.row_stack([_filter, self.bottomConvSample])
            new_filter = new_filter.reshape((1,) + new_filter.shape)

            if len(new_w) == 0:
                new_w = new_filter
            else:
                new_w = np.row_stack([new_w, new_filter])

        bottomConv_Weight = new_w.transpose(2, 3, 1, 0)


        # Apply new Weights in Layers
        topConvWB = [topConv_Weight, topConv_Bias]
        seDense1WB = [seDense1_Weight]
        seDense2WB = [seDense2_Weight]
        bottomConvWB = [bottomConv_Weight, bottomConv_Bias]

        self.filter_num += 1
        self.prepare()

        print('\n========== [Increase After] ============\n')

        print('TopConv Increased >> ', topConv_Weight.shape)
        # print('TopConv Increased >> ', topConv_Weight)
        print('SEDense_In Increased >> ', seDense1_Weight.shape)
        # print('SEDense_In Increased >> ', seDense1_Weight)
        print('SEDense_Out Increased >> ', seDense2_Weight.shape)
        # print('SEDense_Out Increased >> ', seDense2_Weight)
        print('BottomConv Increased >> ', bottomConv_Weight.shape)
        # print('BottomConv Increased >> ', bottomConv_Weight)

        print('Inner Node Num : ', self.SE_inner_node)
        print("")

        self.TopConv = keras.layers.Conv2D(name='SE-TopConv', filters=self.filter_num,
                                           kernel_size=self.kernel_size, padding='same',
                                           activation='relu', weights=topConvWB)
        self.GAP = keras.layers.GlobalAveragePooling2D()
        self.Reshape = keras.layers.Reshape((1, 1, self.filter_num))
        self.SEDense_In = keras.layers.Dense(name='SE-Dense_In', units=self.SE_inner_node,
                                             activation='relu', weights=seDense1WB, use_bias=False)
        self.SEDense_Out = keras.layers.Dense(name='SE-Dense_Out', units=self.filter_num,
                                              activation='sigmoid', kernel_initializer='he_normal',
                                              weights=seDense2WB, use_bias=False)
        self.BottomConv = keras.layers.Conv2D(name='SE-Bottom', filters=self.init_filter,
                                              kernel_size=self.kernel_size, padding='same',
                                              activation='relu', weights=bottomConvWB, trainable=self.bottomTrain)
        self.Scaler = keras.layers.Multiply()

    def getHSEName(self):
        return self.layerName
