"""
@File : train_SubSystem1.py
@Author: Dong Wang
@Date : 2020/5/8
"""

import numpy as np
from core.net import net
from core.net import train
from tensorflow.keras.models import load_model

# File paths
rootPath = 'D:/data/IIS/'
Subsystem1FolderPath = 'Dataset Subsystem 1/'
Subsystem1CSVName = 'Video Data.csv'
Subsystem1CSVConvertedName = 'VideoData2.csv'
Subsystem1VideosPath = 'videos/'
Subsystem1ImagesPath = 'images/'
conditions = ["open_palm", "open_dorsal", "fist_palm", "fist_dorsal",
              "three_fingers_palm", "three_fingers_dorsal"]
fileType = '.webm'
height, width, channel = 240, 320, 3

X_train = np.load(rootPath + Subsystem1FolderPath + 'npy/X_train' + str(height) + '_' + str(width) + '_' + '.npy')
y_train = np.load(rootPath + Subsystem1FolderPath + 'npy/Y_train' + str(height) + '_' + str(width) + '_' + '.npy')


model = net(pretrained_weights='./saved_models/Sub1weights-best.hdf5')

train(model=model, x=X_train, y=y_train, epochs=10, batch_size=4, validation_split=0.25)

# For test
X_test = np.load(rootPath + Subsystem1FolderPath + 'npy/X_test'+str(height)+'_'+str(width)+'_'+'.npy')
y_test = np.load(rootPath + Subsystem1FolderPath + 'npy/Y_test'+str(height)+'_'+str(width)+'_'+'.npy')
trained_model = load_model('./saved_models/Sub1weights-best.hdf5')
trained_model.evaluate(X_test, y_test)
