#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from collections import namedtuple

from keras.preprocessing.image import load_img, img_to_array

ClassItem = namedtuple('ClassItem', ['c', 'path'])


class NNHelp():
    """docstring for NNHelp"""

    def __init__(self, folder, batch_size=16):
        self.textures = ['blanket1', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1', 'floor1', 'floor2', 'grass1', 'linseeds1', 'oatmeal1', 'blanket2', 'lentils1', 'pearlsugar1',
                         'rice1', 'rice2', 'rug1', 'sand1', 'scarf1', 'scarf2', 'screen1', 'seat1', 'seat2', 'sesameseeds1', 'stone1', 'stone2', 'stone3', 'stoneslab1', 'wall1']
        self.classes = len(self.textures)
        self.batch_size = batch_size
        self.__make_paths__(folder)

    def __make_paths__(self, folder):
        ''' make (class, Path) tupple for every element '''
        self.images = []
        folder = Path(folder)
        texturesfolder = [folder / x for x in self.textures]
        for i, j in enumerate(texturesfolder):
            self.images.extend(
                list(
                    map(
                        lambda x: ClassItem(i, j / x), j.iterdir()
                    )
                )
            )

        self.images = np.array(self.images)
        np.random.shuffle(self.images)
        self.lenimg = len(self.images)

    def __imgload__(self, img, size=(224, 224)):
        ''' Load image, return numpy array '''
        return img_to_array(
            load_img(img, target_size=size)
        )

    def __gengen__(self, i):
        ''' generator for generator,
            yields batch_size of elements,
            or lesser on last one '''
        bs = self.batch_size
        if i + bs >= self.lenimg:
            bs = self.lenimg - i
        yield from self.images[i:i + bs]

    def gen(self):
        ''' generator for model trainig,
            yields arrays of batch_size elements,
            or lesser on last one '''
        q = 0
        while True:
            inputs = []
            targets = []
            for i in self.__gengen__(q):
                inputs.append(self.__imgload__(i.path))
                output = np.zeros(self.classes)
                output[int(i.c)] = 1
                targets.append(output)

            yield np.array(inputs), np.array(targets)

            q += self.batch_size
            if q + self.batch_size >= self.lenimg + self.batch_size:
                q = 0
                np.random.shuffle(self.images)
