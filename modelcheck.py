from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np
import os

import argparse
parser = argparse.ArgumentParser(description='Test neural network')
parser.add_argument('-b', '--batchsize', dest='batch_size', action='store',
                    type=int, default=16,
                    help='Batch size')
parser.add_argument('-f', '--folder', dest='folder', action='store',
                    default='ds/cktd/',
                    help='Folder with pictures')
parser.add_argument('-m', '--model', dest='modelfile', action='store',
                    default='models/textureVGGmodel',
                    help='Model filename')
parser.add_argument('-p', '--predict', dest='imgfile', action='store',
                    default=None,
                    help='Image file to predict image')

args = parser.parse_args()

modelfile = args.modelfile
batch_size = args.batch_size
imgfile = args.imgfile
folder = args.folder

model = load_model(modelfile)


def predictOneImage(img):
    # load an image from file
    image = load_img(img, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    yhat = model.predict(image)
    return yhat


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

# textures = ['blanket1', 'canvas1']
postfix = '_test'

images = []
for i, j in enumerate(textures):
    images.extend(
        list(map(lambda x: (i, f'{j}{postfix}/{x}'), os.listdir(f'{folder}/{j}{postfix}'))))

images = np.array(images)
np.random.shuffle(images)


def gengen(i, batch_size):
    bs = batch_size
    if i + bs > len(images):
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
        yield np.array(inputs), np.array(targets)
        q += batch_size
        if q + batch_size >= len(images) + batch_size:
            q = 0


def eg():
    return model.evaluate_generator(gen(batch_size),
                                    steps=int(len(images) / batch_size),
                                    # steps=100,
                                    )


if __name__ == '__main__':
    if imgfile is not None:
        print(predictOneImage(imgfile))
    else:
        print(eg())
