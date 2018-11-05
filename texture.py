from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.callbacks import CSVLogger, EarlyStopping, TerminateOnNaN, ModelCheckpoint, LambdaCallback

import numpy as np
import os
from time import time
import argparse

from helpNN import NNHelp


def mkIfNeed(*folders):
    ''' Проверить существует ли каталог, состоящий из folders,
        создать если нет, вернуть путь '''
    if not os.path.exists(os.path.join(*folders)):
        os.makedirs(os.path.join(*folders))
    return os.path.join(*folders)


def _parseArgs():
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--net', dest='net', action='store',
                        required=True,
                        choices=['VGG', 'RN', 'I', 'X'],
                        help='Choose Neural Network. VGG16, ResNet50, Inception, Xception as %(choices)s')
    parser.add_argument('-b', '--batchsize', dest='batch_size', action='store',
                        type=int, default=16,
                        help='Batch size')
    parser.add_argument('-f', '--train', dest='train', action='store',
                        default=os.path.join('ds', 'cktd', 'train'),
                        help='Folder with train pictures')
    parser.add_argument('--test', dest='test', action='store',
                        default=os.path.join('ds', 'cktd', 'test'),
                        help='Folder with test pictures')
    parser.add_argument('-m', '--model', dest='modelfile', action='store',
                        default=os.path.join('models', 'textureModel'),
                        help='Name of the file where model will be stored. To this name will be added "-$numofclasses$_b$batch_size$')
    parser.add_argument('-l', '--log', dest='logfile', action='store',
                        default=os.path.join('logs', 'training.csv'),
                        help='logfile name')

    return parser.parse_args()


if __name__ == '__main__':
    args = _parseArgs()
    batch_size = args.batch_size
    train = args.train
    test = args.test
    nnhelp = NNHelp(train, batch_size)
    nnhelptest = NNHelp(test, batch_size)

    modelfile = args.modelfile
    logfile = args.logfile + '.csv'

    classes = nnhelp.classes

    print('#' * 40, f'# {classes} classes, {batch_size} batch_size, {args.net}',
          f'# Get train pictures from {train}',
          f'# {modelfile}', f'# {logfile}', '#' * 40, sep=os.linesep)

    if args.net == 'VGG':
        model = VGG16(include_top=True, weights=None, classes=classes)
        print('###### Using VGG')
        name = 'VGG'
    if args.net == 'RN':
        model = ResNet50(include_top=True, weights=None, classes=classes)
        print('###### Using RN')
        name = 'ResNet'
    if args.net == 'I':
        model = InceptionV3(include_top=True, weights=None, classes=classes)
        print('###### Using I')
        name = 'Inception'
    if args.net == 'X':
        model = Xception(include_top=True, weights=None, classes=classes)
        print('###### Using X')
        name = 'Xception'

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    start = time()
    diff = 0
    with open(logfile, 'w') as f:
        f.write('name,epoch,time,acctrain,losstrain,acctest,losstest\n')

    for epoch in range(50):
        hist = model.fit_generator(nnhelp.gen(),
                                   steps_per_epoch=int(
                                       nnhelp.lenimg / batch_size),
                                   initial_epoch=epoch,
                                   epochs=epoch + 1)
        diffstart = time()
        res = model.evaluate_generator(nnhelp.gen(),
                                       steps=int(nnhelp.lenimg / batch_size))
        diff += time() - diffstart

        print(name, epoch, time() - diff - start,
              hist.history['acc'][0], hist.history['loss'][0],
              res[1], res[0], sep=', ')

        with open(logfile, 'a') as f:
            print(name, epoch, time() - diff - start,
                  hist.history['acc'][0], hist.history['loss'][0],
                  res[1], res[0], sep=',', file=f)

    finish = time()
    print(f'### Time: {finish-start}')
    with open(os.path.join('logs', 'time.log'), 'a') as f:
        print(f'{modelfile} : {finish-start}', file=f)

    model.save(modelfile)
