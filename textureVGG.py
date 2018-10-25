from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16

from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN, ModelCheckpoint

import numpy as np
import os
from time import time

import argparse
parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('-b', '--batchsize', dest='batch_size', action='store',
                    type=int, default=16,
                    help='Batch size')
parser.add_argument('-f', '--folder', dest='folder', action='store',
                    default='ds/cktd/',
                    help='Folder with pictures')
parser.add_argument('-m', '--model', dest='modelfile', action='store',
                    default='models/textureVGGmodel',
                    help='Name of the file where model will be stored')
parser.add_argument('-l', '--log', dest='logfile', action='store',
                    default='logs/training.csv',
                    help='logfile name')

args = parser.parse_args()

modelfile = args.modelfile
logfile = args.logfile

batch_size = args.batch_size
folder = args.folder

textures = ['blanket1',
            'canvas1',
            'ceiling1',
            'ceiling2',
            'cushion1',
            'floor1',
            'floor2',
            'grass1',
            'linseeds1',
            'oatmeal1']

textures = ['blanket1', 'blanket2', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1', 'floor1', 'floor2', 'grass1', 'lentils1', 'linseeds1', 'oatmeal1', 'pearlsugar1', 'rice1', 'rice2', 'rug1', 'sand1', 'scarf1', 'scarf2', 'screen1', 'seat1', 'seat2', 'sesameseeds1', 'stone1', 'stone2', 'stone3', 'stoneslab1', 'wall1']

classes = len(textures)

model = VGG16(include_top=True, weights=None, classes=classes)

images = []
for i, j in enumerate(textures):
    images.extend(
        list(map(lambda x: (i, j + '/' + x), os.listdir(folder + j))))

images = np.array(images)
np.random.shuffle(images)


def gengen(i, batch_size):
    bs = batch_size
    if i + bs >= len(images):
        bs = len(images) - i
    yield from images[i:i + bs]


def imgload(img):
    # load an image from file
    image = load_img(img, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    return image


def gen(batch_size):
    q = 0
    while True:
        inputs = []
        targets = []
        for i in gengen(q, batch_size):
            inputs.append(imgload(folder + i[1]))
            output = np.zeros(classes)
            output[int(i[0])] = 1
            targets.append(output)
        yield np.array(inputs), np.array(targets)
        q += batch_size
        if q + batch_size >= len(images) + batch_size:
            q = 0
            np.random.shuffle(images)


# model.summary()

callbacks = [
    CSVLogger(filename=logfile),
    # EarlyStopping(monitor='val_acc', mode='max', min_delta=0.01, patience=5, baseline=0.95, verbose=1),
    # EarlyStopping(monitor='acc', mode='auto', patience=10, verbose=1),
    TerminateOnNaN(),
    ModelCheckpoint('models/modelCheckpoint', monitor='acc', verbose=1, save_best_only=True, mode='auto', period=10)
]

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

start = time()

model.fit_generator(gen(batch_size=batch_size),
                    steps_per_epoch=int(len(images) / batch_size),
                    epochs=100,
                    callbacks=callbacks)

finish = time()
print(f'### Time: {finish-start}')

model.save(modelfile)
