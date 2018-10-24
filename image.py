import cv2
import os
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Prepare images')
parser.add_argument('--crop', dest='crop', action='store_true',
                    help='Crop images in 4 pieces')
parser.add_argument('--folder', dest='folder', action='store',
                    default='ds/ktd',
                    help='Folder with original images')
parser.add_argument('--newfolder', dest='newfolder', action='store',
                    default='ds/cktd',
                    help='Folder with cropped images images')
parser.add_argument('--septest', dest='septest', action='store_true',
                    help='Choose 1/4 images as test data and move them in the other folder')

args = parser.parse_args()


def cropInFour(folder, newfolder):
    for i in os.listdir(folder):
        os.mkdir(f'{newfolder}/{i}')
        for j in os.listdir(f'{folder}/{i}'):
            img = cv2.imread(folder + '/' + i + '/' + j)
            img1, img2 = np.array_split(img, 2) 
            img11, img12 = np.array_split(img1, 2, axis=1)
            img21, img22 = np.array_split(img2, 2, axis=1)
            cv2.imwrite(newfolder + '/' + i + '/' + '11_' + j, img11)
            cv2.imwrite(newfolder + '/' + i + '/' + '12_' + j, img12)
            cv2.imwrite(newfolder + '/' + i + '/' + '21_' + j, img21)
            cv2.imwrite(newfolder + '/' + i + '/' + '22_' + j, img22)
            print(f"{i}/11_{j}\n\
                    {i}/12_{j}\n\
                    {i}/21_{j}\n\
                    {i}/22_{j}")


def moveTestImages(folder):
    for i in os.listdir(folder):
        os.mkdir(f'{folder}/{i}_test')
        listdir = np.array(os.listdir(f'{folder}/{i}'))
        np.random.shuffle(listdir)
        for j in listdir[:int(len(listdir) / 4)]:
            os.rename(f'{folder}/{i}/{j}', f'{folder}/{i}_test/{j}')
            print(f'{folder}/{i}/{j}', f'{folder}/{i}_test/{j}')


if __name__ == '__main__':
    # folder = '/home/dave/mag/tf/ds/ktd'
    # newfolder = '/home/dave/mag/tf/ds/cktd'
    folder = args.folder
    newfolder = args.newfolder

    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
        print(f'Creating {newfolder}')

    if args.crop:
        print('Croping images')
        cropInFour(folder, newfolder)

    if args.septest:
        print('Moving images')
        moveTestImages(newfolder)
