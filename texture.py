from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN, ModelCheckpoint

import numpy as np
import os
from time import time
import argparse


def mkIfNeed(*folders):
    ''' Проверить существует ли каталог, состоящий из folders, создать если нет, вернуть путь '''
    if not os.path.exists(os.path.join(*folders)):
        os.makedirs(os.path.join(*folders))
    return os.path.join(*folders)


parser = argparse.ArgumentParser(description='Train neural network')
parser.add_argument('--net', dest='net', action='store',
                    required=True,
                    choices=['VGG', 'RN', 'I', 'X'],
                    help='Choose Neural Network. VGG16, ResNet50, Inception, Xception as %(choices)s')
parser.add_argument('-b', '--batchsize', dest='batch_size', action='store',
                    type=int, default=16,
                    help='Batch size')
parser.add_argument('-f', '--folder', dest='folder', action='store',
                    default=os.path.join('ds', 'cktd', 'train'),
                    help='Folder with train pictures')
parser.add_argument('-m', '--model', dest='modelfile', action='store',
                    default=os.path.join('models', 'textureModel'),
                    help='Name of the file where model will be stored. To this name will be added "-$numofclasses$_b$batch_size$')
parser.add_argument('-l', '--log', dest='logfile', action='store',
                    default=os.path.join('logs', 'training.csv'),
                    help='logfile name')
parser.add_argument('-10', dest='ten', action='store_true',
                    help='Run on 10 classes. (hardcoded)')

args = parser.parse_args()

# modelfile = args.modelfile
# logfile = args.logfile

batch_size = args.batch_size
folder = args.folder

nameappend = '-' + ('10' if args.ten else '28') + \
    '_b' + str(args.batch_size)

modelfile = args.modelfile + nameappend
logfile = args.logfile + nameappend + '.csv'

if args.ten:
    textures = ['blanket1', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1',
                'floor1', 'floor2', 'grass1', 'linseeds1', 'oatmeal1']
else:
    textures = ['blanket1', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1', 'floor1', 'floor2', 'grass1', 'linseeds1', 'oatmeal1', 'blanket2', 'lentils1', 'pearlsugar1',
                'rice1', 'rice2', 'rug1', 'sand1', 'scarf1', 'scarf2', 'screen1', 'seat1', 'seat2', 'sesameseeds1', 'stone1', 'stone2', 'stone3', 'stoneslab1', 'wall1']

classes = len(textures)
print('#' * 40, f'# {classes} classes, {batch_size} batch_size, {args.net}',
      f'# Get train pictures from {folder}',
      f'# {modelfile}', f'# {logfile}', '#' * 40, sep=os.linesep)

# model = VGG16(include_top=True, weights=None, classes=classes)
# model = ResNet50(include_top=True, weights=None, classes=classes)
# model = InceptionV3(include_top=True, weights=None, classes=classes)
# model = Xception(include_top=True, weights=None, classes=classes)

if args.net == 'VGG':
    model = VGG16(include_top=True, weights=None, classes=classes)
    print('###### Using VGG')
if args.net == 'RN':
    model = ResNet50(include_top=True, weights=None, classes=classes)
    print('###### Using RN')
if args.net == 'I':
    model = InceptionV3(include_top=True, weights=None, classes=classes)
    print('###### Using I')
if args.net == 'X':
    model = Xception(include_top=True, weights=None, classes=classes)
    print('###### Using X')


images = []
for i, j in enumerate(textures):
    #images.extend(list(map(lambda x: (i, f'{j}{postfix}/{x}'), os.listdir(f'{folder}/{j}{postfix}'))))
    images.extend(
        list(map(lambda x: (i, os.path.join(j, x)), os.listdir(os.path.join(folder, j)))))

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
            inputs.append(imgload(os.path.join(folder, i[1])))
            output = np.zeros(classes)
            output[int(i[0])] = 1
            targets.append(output)
        yield np.array(inputs), np.array(targets)
        q += batch_size
        if q + batch_size >= len(images) + batch_size:
            q = 0
            np.random.shuffle(images)


# model.summary()
callbacktime = time()

callbacks = [
    CSVLogger(filename=logfile),
    # EarlyStopping(monitor='val_acc', mode='max', min_delta=0.01, patience=5, baseline=0.95, verbose=1),
    # EarlyStopping(monitor='acc', mode='auto', patience=10, verbose=1),
    TerminateOnNaN(),
    # ModelCheckpoint(os.path.join(mkIfNeed('models', 'tmp'), 'mCP') + f'~{args.net}~' + nameappend + '~{epoch:03d}-{acc:.4f}~',
    ModelCheckpoint(os.path.join(mkIfNeed('models', 'tmp'), 'mCP') + f'~{os.path.split(args.modelfile)[1]}~' + nameappend + '~{epoch:03d}-{acc:.4f}',
                    monitor='acc',
                    verbose=1, save_best_only=True, mode='auto', period=3)
]

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

start = time()

model.fit_generator(gen(batch_size=batch_size),
                    steps_per_epoch=int(len(images) / batch_size),
                    epochs=20,
                    callbacks=callbacks)

finish = time()
print(f'### Time: {finish-start}')
with open(os.path.join('logs', 'time.log'), 'a') as f:
    print(f'{modelfile} : {finish-start}', file=f)

model.save(modelfile)
