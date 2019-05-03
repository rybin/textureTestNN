import cv2
import os
import numpy as np
import argparse


def _parseArgs():
    parser = argparse.ArgumentParser(description='''Prepare images.

Default structure:
Folders with original images: ds/ktd/, [ds/ktd/blanket1, ...];
Folders with cropped images:  ds/cktd/x4, [ds/cktd/x4/blanket1, ...];
Folders with train images:    ds/cktd/train, [ds/cktd/train/blanket1, ...];
Folders with test images:     ds/cktd/test, [ds/cktd/test/blanket1, ...];''',
                                     formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers()

    crop_parser = subparsers.add_parser('crop', help='Crop images in 4 parts')
    crop_parser.add_argument('--folder', dest='folder', action='store',
                             default=os.path.join('ds', 'ktd'),
                             help='Folder with folders original images')
    crop_parser.add_argument('--cropped', dest='cropped', action='store',
                             default=os.path.join('ds', 'cktd', 'x4'),
                             help='Folder with folders with cropped images')
    crop_parser.set_defaults(func=crop)

    move_parser = subparsers.add_parser(
        'move', help='Move train from test data. randomly. This operation move images to other folder rather than copy them.')
    move_parser.add_argument('--cropped', dest='cropped', action='store',
                             default=os.path.join('ds', 'cktd', 'x4'),
                             help='Folder with folders with cropped images')
    move_parser.add_argument('--train', dest='train', action='store',
                             default=os.path.join('ds', 'cktd', 'train'),
                             help='Folder with train data')
    move_parser.add_argument('--test', dest='test', action='store',
                             default=os.path.join('ds', 'cktd', 'test'),
                             help='Folder with test data')
    move_parser.add_argument('--part', type=str, dest='part', action='store',
                             default='4.0',
                             help='Part of dataset set as test data. 1/PART of original dataset. Other set as train data.')
    move_parser.set_defaults(func=move)

    return parser.parse_args()


def mkIfNeed(*folders):
    ''' Проверить существует ли каталог, состоящий из folders, создать если нет, вернуть путь '''
    if not os.path.exists(os.path.join(*folders)):
        # os.mkdir(os.path.join(*folders))
        os.makedirs(os.path.join(*folders))
    return os.path.join(*folders)


def cropInFour(folder, newfolder):
    x4f = mkIfNeed(newfolder)

    # По всем каталогам с картинками
    for i in os.listdir(folder):
        imgx4f = mkIfNeed(x4f, i)
        # По всем картинкам
        for j in os.listdir(os.path.join(folder, i)):
            img = cv2.imread(os.path.join(folder, i, j))
            # Разбить на 4 части
            img1, img2 = np.array_split(img, 2)
            img11, img12 = np.array_split(img1, 2, axis=1)
            img21, img22 = np.array_split(img2, 2, axis=1)
            # Записать 4 части в отдельный каталог
            cv2.imwrite(os.path.join(imgx4f, f'11_{j}'), img11)
            cv2.imwrite(os.path.join(imgx4f, f'12_{j}'), img12)
            cv2.imwrite(os.path.join(imgx4f, f'21_{j}'), img21)
            cv2.imwrite(os.path.join(imgx4f, f'22_{j}'), img22)

            print(f"#    11_{j}", f"##   12_{j}",
                  f"###  21_{j}", f"#### 22_{j}", sep=os.linesep)
        print(f"Finish {i}")


def crop(args):
    cropInFour(args.folder, args.cropped)


def movePart(folder, train, test, part):
    # По всем каталогам с картинками
    for k in os.listdir(folder):
        tn = mkIfNeed(train, k)
        tt = mkIfNeed(test, k)
        # построить массив с именами
        listdir = np.array(os.listdir(os.path.join(folder, k)))
        # перемешать
        np.random.shuffle(listdir)
        # part от массива переместить в test
        for j in listdir[:int(len(listdir) / part)]:
            os.rename(os.path.join(folder, k, j),
                      os.path.join(tt, j))
            print('Test:', os.path.join(folder, k, j), '>', os.path.join(tt, j))
        # оставшееся в train
        for j in listdir[int(len(listdir) / part):]:
            os.rename(os.path.join(folder, k, j),
                      os.path.join(tn, j))
            print('Train:', os.path.join(folder, k, j), '>', os.path.join(tn, j))


def moveNumber(folder, train, test, part_train, part_test):
    # По всем каталогам с картинками
    for k in os.listdir(folder):
        tn = mkIfNeed(train, k)
        tt = mkIfNeed(test, k)
        # построить массив с именами
        listdir = np.array(os.listdir(os.path.join(folder, k)))
        if (len(listdir) / 2 < part_train) or \
           (len(listdir) / 2 < part_test):
            print(f'{part_train} | {part_test} > {len(listdir)}')
            return -1
        # перемешать
        np.random.shuffle(listdir)
        # первые part_test в test
        for j in listdir[:part_test]:
            os.rename(os.path.join(folder, k, j),
                      os.path.join(tt, j))
            print('Test:', os.path.join(folder, k, j), '>', os.path.join(tt, j))
        # последние part_train в train
        for j in listdir[(len(listdir) - part_train):]:
            os.rename(os.path.join(folder, k, j),
                      os.path.join(tn, j))
            print('Train:', os.path.join(folder, k, j), '>', os.path.join(tn, j))


def moveTestImages(folder, _train, _test, part):
    # Создание train и test катологов
    train = mkIfNeed(_train)
    test = mkIfNeed(_test)

    if ':' in part:
        part_train, part_test = part.split(':')
        part_train = int(part_train)
        part_test = int(part_test)
        moveNumber(folder, train, test, part_train, part_test)
    else:
        part = float(part)
        if part <= 1:
            print('PART cannot be equal or leser than 1')
            return -1
        movePart(folder, train, test, part)


def move(args):
    moveTestImages(args.cropped, args.train, args.test, args.part)


if __name__ == '__main__':
    args = _parseArgs()
    args.func(args)
