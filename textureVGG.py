from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16

from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN

import numpy as np
import os

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

model = VGG16(include_top=True, weights=None, classes=10)

modelfile = args.modelfile
logfile = args.logfile

batch_size = args.batch_size
folder = args.folder

# batch_size = 16
# folder = '/home/dave/mag/tf/ds/cktd/'

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
            output = np.zeros(len(textures))
            output[int(i[0])] = 1
            targets.append(output)
        # print(np.array(inputs).shape)
        # print(np.array(targets).shape)
        yield np.array(inputs), np.array(targets)
        q += batch_size
        if q + batch_size >= len(images) + batch_size:
            q = 0
            np.random.shuffle(images)


# model.summary()

callbacks = [
    CSVLogger(filename=logfile),
    EarlyStopping(monitor='val_acc', patience=5, mode='max', baseline=0.95),
    TerminateOnNaN()
]

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

model.fit_generator(gen(batch_size=batch_size),
                    steps_per_epoch=int(len(images) / batch_size),
                    epochs=100,
                    callbacks=callbacks)

model.save(modelfile)
