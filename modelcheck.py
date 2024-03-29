from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np
import os
import argparse


def _parseArgs():
    parser = argparse.ArgumentParser(description='Test neural network')
    parser.add_argument('-b', '--batchsize', dest='batch_size', action='store',
                        type=int, default=16,
                        help='Batch size')
    parser.add_argument('-f', '--folder', dest='folder', action='store',
                        default=os.path.join('ds', 'cktd', 'test'),
                        help='Folder with test pictures')
    parser.add_argument('-m', '--model', dest='modelfile', action='store',
                        default=os.path.join('models', 'textureModel'),
                        help='Model filename')
    parser.add_argument('-10', dest='ten', action='store_true',
                        help='Test on 10 classes. (hardcoded)')

    subparsers = parser.add_subparsers()

    predict_parser = subparsers.add_parser(
        'predict', help='Predict specified images')
    predict_parser.add_argument('images', nargs='+',
                                help='Filenames of images. One or multiple.')
    predict_parser.set_defaults(func=predictImages)

    check_parser = subparsers.add_parser(
        'check', help='Check loss and accuracy of NN')
    check_parser.set_defaults(func=checkAll)

    report_parser = subparsers.add_parser(
        'report', help='Test NN and create report in numpy arrays')
    report_parser.add_argument('--report', dest='reportfile', action='store',
                               default='nn-report',
                               help='Specify file to save report')
    report_parser.set_defaults(func=getTruePred)

    return parser.parse_args()


def loadImage(img):
    image = load_img(img, target_size=(224, 224))
    image = img_to_array(image)
    return image


def genOneImage(images):
    for i in images:
        img = loadImage(i)
        yield img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))


def predictImages(args):
    images = args.images
    return model.predict_generator(genOneImage(images),
                                   steps=len(images))


def loadImageNameGlobal(ten):
    global textures
    global images

    if ten:
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
    else:
        textures = ['blanket1', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1', 'floor1', 'floor2', 'grass1', 'linseeds1', 'oatmeal1', 'blanket2', 'lentils1', 'pearlsugar1',
                    'rice1', 'rice2', 'rug1', 'sand1', 'scarf1', 'scarf2', 'screen1', 'seat1', 'seat2', 'sesameseeds1', 'stone1', 'stone2', 'stone3', 'stoneslab1', 'wall1']

    images = []
    for i, j in enumerate(textures):
        images.extend(
            list(map(lambda x: (i, os.path.join(j, x)), os.listdir(os.path.join(folder, j)))))

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
            inputs.append(imgload(os.path.join(folder, i[1])))
            output = np.zeros(len(textures))
            output[int(i[0])] = 1
            targets.append(output)
        yield np.array(inputs), np.array(targets)
        q += batch_size
        if q + batch_size >= len(images) + batch_size:
            q = 0


def checkAll(args):
    loadImageNameGlobal(args.ten)
    res = model.evaluate_generator(gen(batch_size),
                                   steps=int(len(images) / batch_size),
                                   # steps=100,
                                   )
    with open(os.path.join('logs', 'check.log'), 'a') as f:
        print(f'{modelfile},{res[1]},{res[0]}', file=f)
    return res


def genPred():
    return model.predict_generator(gen(batch_size),
                                   steps=int(len(images) / batch_size),
                                   )


def getTruePred(args):
    loadImageNameGlobal(args.ten)
    report = images.T[0].astype(int), np.array([x.argmax() for x in genPred()])
    np.save(args.reportfile, report)
    return f'Saved to {args.reportfile}'


if __name__ == '__main__':

    args = _parseArgs()
    print(args)

    global modelfile
    global batch_size
    global folder
    global model

    modelfile = args.modelfile
    batch_size = args.batch_size
    folder = args.folder

    print('#' * 40, f'# {10 if args.ten else 28} classes, {batch_size} batch_size',
          f'# Get test pictures from {folder}',
          f'# {modelfile}', '#' * 40, sep=os.linesep)

    model = load_model(modelfile)
    # loadImageNameGlobal(args.ten)
    result = args.func(args)
    print(result)
