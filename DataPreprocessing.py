"""
@File : DataPreprocessing.py
@Author: Dong Wang
@Date : 2020/5/16
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from tqdm.auto import trange, tqdm

import sys
import cv2
from PIL import Image
import glob
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


rootPath = 'D:/data/IIS/'
Subsystem1FolderPath = 'Dataset Subsystem 1/'
Subsystem1CSVName = 'Video Data.csv'
Subsystem1CSVConvertedName = 'VideoData2.csv'
Subsystem1VideosPath = 'videos/'
Subsystem1ImagesPath = 'images/'
extrnalCSVName = 'Dong_Wang_9446_annotations.csv'
extrnalID = 446
conditions = ["open_palm", "open_dorsal", "fist_palm", "fist_dorsal",
              "three_fingers_palm", "three_fingers_dorsal"]
fileType = '.webm'

'''
Find all videos
'''
videoPaths = []
personIDs = []
for personID in glob.glob(rootPath + Subsystem1FolderPath + Subsystem1VideosPath + '*/'):
    personID = personID.replace("\\", "/")
    personIDs.append(personID)
    for con in conditions:
        video = personID + con + fileType
        videoPaths.append(video)

numPerson = len(personIDs)
numVideos = len(videoPaths)

print("[DataLoader]: The number of person : ", numPerson)
print("[DataLoader]: The number of video : ", numVideos)


def videoHandler(videofile=None, condition=None, outputPath=None, savetoDisk=False):
    if videofile is None:
        return 0
    #     print(videofile)
    if videofile.find('palm') > 0:
        prex = 'palm'
    elif videofile.find('dorsal') > 0:
        prex = 'dorsal'

    Sides = ['palm', 'dorsal']

    imageFilePaths = []

    input_video = cv2.VideoCapture(videofile)
    ret, frame = input_video.read()
    i = 0
    while ret:
        ret, frame = input_video.read()

        if not ret:
            continue
        if savetoDisk:
            cv2.imwrite(outputPath + condition + '_' + str(i) + '.jpg', frame)
        imageFilePaths.append(outputPath + condition + '_' + str(i) + '.jpg')
        i += 1
    input_video.release()
    return np.array(imageFilePaths), i


height, width, channel = 240, 320, 3
# rootPath+Subsystem1FolderPath+Subsystem1CSVwithExtrnalDataName


"""
Save all video to images and/or load images to memory
"""

is_first_video = True
for personID in tqdm(personIDs, desc='Process persons'):
    for con in conditions:
        outputPath = personID.replace('videos', 'images')
        imageFilePaths, num_frame = videoHandler(videofile=personID + con + fileType, condition=con,
                                                 outputPath=outputPath, savetoDisk=True)
        # Load img to an array
        imgs = np.ndarray((num_frame, height, width, 3), dtype=np.uint8)
        for i in range(num_frame):
            img = load_img(imageFilePaths[i], color_mode="rgb")
            img = img.resize((width, height), Image.LANCZOS)
            img = img_to_array(img)
            imgs[i] = img
        if is_first_video:
            last_imgs = imgs
            last_imageFilePaths = imageFilePaths
            is_first_video = False

        else:
            last_imgs = np.concatenate((last_imgs, imgs), axis=0)
            last_imageFilePaths = np.concatenate((last_imageFilePaths, imageFilePaths), axis=0)

data = last_imgs
data_paths = last_imageFilePaths
print("[DataLoader] There are total ", data.shape[0], "images")


def savetoDisk(data):
    data = data.astype('float32')
    data /= 255
    np.save(rootPath + Subsystem1FolderPath + 'npy/imgs' + str(height) + '_' + str(width) + '_' + '.npy', data)
    print('Save npy file to ' + rootPath + Subsystem1FolderPath + 'npy/imgs' + str(height) + '_' + str(
        width) + '_' + '.npy')
    print("[DataLoader] It costs ", sys.getsizeof(data) / 1024 ** 3, "GB to store")


print("[DataLoader] It costs ", sys.getsizeof(data) / 1024 ** 3, "GB in memory")


"""
Load anatation to memory
"""

anatations = pd.read_csv(rootPath + Subsystem1FolderPath + Subsystem1CSVName)

anatationsTransfered = anatations

for col in anatationsTransfered.columns.values[5:]:
    anatationsTransfered[col] = anatationsTransfered[col] / 2
    anatationsTransfered[col].astype('category')

anatationsTransfered.to_csv(rootPath + Subsystem1FolderPath + Subsystem1CSVConvertedName, index=False)

ii = []
dataResorted = np.ndarray((anatationsTransfered.shape[0], height, width, 3), dtype=np.float32)
for k in range(len(last_imageFilePaths)):
    path = last_imageFilePaths[k]
    prex = path[path.find('images') + 7:]
    frame = int(prex[prex.rfind('_') + 1:prex.find('.jpg')])
    source = prex[:prex.rfind('_')] + '.webm'
    #     print(prex, source,frame)
    query = anatationsTransfered[
        (anatationsTransfered.source == source) & (anatationsTransfered.frame == frame)].index.values
    if len(query) == 0:
        print("There is no annation for ", prex)
        continue
    i = query[0]
    ii.append(i)
    dataResorted[i] = data[k]

savetoDisk(dataResorted)

print("[DataLoader] It costs ", sys.getsizeof(dataResorted) / 1024 ** 3, "GB in memory")

data = np.load('D:/data/IIS/Dataset Subsystem 1/npy/imgs240_320_.npy')

anatationsTransfered = pd.read_csv(rootPath + Subsystem1FolderPath + Subsystem1CSVConvertedName)

anatationsArray = anatationsTransfered[anatationsTransfered.columns.values[5:]].values
anatationsArray /= 255
np.save(rootPath + Subsystem1FolderPath + 'npy/anatations' + str(height) + '_' + str(width) + '_' + '.npy',
        anatationsArray)

gestures = anatationsTransfered[anatationsTransfered.columns.values[3]] + '_' + anatationsTransfered[
    anatationsTransfered.columns.values[4]]
gestureArray = gestures.values

"""
End2end model data
"""

X_train, X_test, y_train, y_test = train_test_split(data, gestureArray, test_size=0.20, random_state=42, shuffle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

np.save(rootPath + Subsystem1FolderPath + 'npy/End2endX_train' + str(height) + '_' + str(width) + '_' + '.npy', X_train)
np.save(rootPath + Subsystem1FolderPath + 'npy/End2endY_train' + str(height) + '_' + str(width) + '_' + '.npy', y_train)
np.save(rootPath + Subsystem1FolderPath + 'npy/End2endX_test' + str(height) + '_' + str(width) + '_' + '.npy', X_test)
np.save(rootPath + Subsystem1FolderPath + 'npy/End2endY_test' + str(height) + '_' + str(width) + '_' + '.npy', y_test)

End2endYtrain = np.load(
    rootPath + Subsystem1FolderPath + 'npy/End2endY_train' + str(height) + '_' + str(width) + '_' + '.npy',
    allow_pickle=True)


def str2OneHot(data):
    labels = np.unique(data)
    for i in range(len(labels)):
        data[data == labels[i]] = i
    return to_categorical(data)


y_trainCategorical = str2OneHot(y_train)
y_testCategorical = str2OneHot(y_test)

np.save(
    rootPath + Subsystem1FolderPath + 'npy/End2endY_trainCategorical' + str(height) + '_' + str(width) + '_' + '.npy',
    y_trainCategorical)
np.save(
    rootPath + Subsystem1FolderPath + 'npy/End2endY_testCategorical' + str(height) + '_' + str(width) + '_' + '.npy',
    y_testCategorical)

"""
Generate a random gestures video from original data sets
"""

I = np.random.choice(last_imageFilePaths.shape[0], size=2400, replace=False)
selectedImages = last_imageFilePaths[I]

output_video = cv2.VideoWriter(rootPath + Subsystem1FolderPath + Subsystem1VideosPath + "test.webm", cv2.CAP_ANY,
                               cv2.VideoWriter_fourcc(*"VP80"), 24, (320, 240))

for im in selectedImages:
    image = cv2.imread(im)
    image = cv2.resize(image, (320, 240))
    for j in range(6):
        output_video.write(image)
output_video.release()

"""
 Data for Sub-system 1
"""

X_train, X_test, y_train, y_test = train_test_split(data, anatationsArray, test_size=0.20, random_state=42,
                                                    shuffle=True)

np.save(rootPath + Subsystem1FolderPath + 'npy/X_train' + str(height) + '_' + str(width) + '_' + '.npy', X_train)
np.save(rootPath + Subsystem1FolderPath + 'npy/Y_train' + str(height) + '_' + str(width) + '_' + '.npy', y_train)
np.save(rootPath + Subsystem1FolderPath + 'npy/X_test' + str(height) + '_' + str(width) + '_' + '.npy', X_test)
np.save(rootPath + Subsystem1FolderPath + 'npy/Y_test' + str(height) + '_' + str(width) + '_' + '.npy', y_test)
