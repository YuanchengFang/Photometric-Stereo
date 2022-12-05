import cv2
import numpy as np
import os
import argparse
from utils import *

# location of pmsData dir
DATA_DIR = './Homework 1/pmsData'

# save results in dir:
RESULT_DIR = './Homework 1/results'

# tasks
TASKS = ['bear', 'buddha', 'cat', 'pot']


def lambert(task, imgs, directions):
    imgs = imgs.reshape(96, -1)

    # Least Square Estimation
    result = np.linalg.lstsq(a=directions, b=imgs, rcond=-1)
    result = result[0]

    # RESHAPE
    result = result.reshape((3, 512, 612)).transpose((1, 2, 0))

    # NORMALIZE
    norm = np.linalg.norm(result, axis=2, keepdims=True)
    normal = np.divide(result, norm, where=result!=0)
    result = (normal + 1) / 2

    # SAVE normal
    result = result * (2**16-1)
    result = result.astype(np.uint16)
    result = np.flip(result, axis=2)
    dirname = os.path.join(RESULT_DIR, 'lambert')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = os.path.join(dirname, './normal_%s.png'%task)
    cv2.imwrite(filename, result)

    # SAVE albedo
    norm = norm.reshape(512, 612) / norm.max() * 255
    norm = norm.astype(np.uint8)
    filename = os.path.join(dirname, './albedo_%s.png'%task)
    cv2.imwrite(filename, norm)

    # RE-RENDER
    '''
    light direction: (0, 0, 1)
    '''
    light_intensities = np.array([0.362, 1.2876, 1.886])
    img = normal[:, :, 2:3] * np.expand_dims(norm, 2).astype(np.float32) / 255 * light_intensities
    if img.max() > 1:
        img = img / img.max()
    img = img * (2**16-1)
    img = img.astype(np.uint16)
    img = np.flip(img, axis=2)
    
    filename = os.path.join(dirname, './re-render_%s.png'%task)
    cv2.imwrite(filename, img)


def lambert_improved(task, imgs, directions, mask, rate=0.25):
    N, H, W = imgs.shape
    normals = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            if mask[i][j] == 0: continue

            cut = int(rate * N)

            # SORT
            pixels = sorted([(imgs[k, i, j], k) for k in range(N)])

            # CUT shadow and bright with rate * N
            pixels = pixels[cut:N-cut]
            
            # Least Square Estimation
            pts, directs = np.array([p[0] for p in pixels]), np.array([directions[p[1]] for p in pixels])
            result = np.linalg.lstsq(a=directs, b=pts, rcond=-1)
            result = result[0]
            normals[i, j, :] = result
    
    result = normals
    norm = np.linalg.norm(result, axis=2, keepdims=True)
    normal = np.divide(result, norm, where=result!=0)
    result = (normal + 1) / 2

    # SAVE normal
    result = result * (2**16-1)
    result = result.astype(np.uint16)
    result = np.flip(result, axis=2)
    dirname = os.path.join(RESULT_DIR, 'lambert-improved')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filename = os.path.join(dirname, './normal_%s.png'%task)
    cv2.imwrite(filename, result)

    # SAVE albedo
    norm = norm.reshape(512, 612) / norm.max() * 255
    norm = norm.astype(np.uint8)
    filename = os.path.join(dirname, './albedo_%s.png'%task)
    cv2.imwrite(filename, norm)

    # RE-RENDER
    '''
    light direction: (0, 0, 1)
    '''
    light_intensities = np.array([0.362, 1.2876, 1.886])
    img = normal[:, :, 2:3] * np.expand_dims(norm, 2).astype(np.float32) / 255 * light_intensities
    if img.max() > 1:
        img = img / img.max()
    img = img * (2**16-1)
    img = img.astype(np.uint16)
    img = np.flip(img, axis=2)
    filename = os.path.join(dirname, './re-render_%s.png'%task)
    cv2.imwrite(filename, img)

    
            


            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default='all')
    parser.add_argument('--improve', action='store_true', default=False)

    args = parser.parse_args()

    if args.name.lower() == 'all':
        tasks = TASKS
    elif args.name.lower() in TASKS:
        tasks = [args.name.lower()]
    else:
        raise AttributeError('%s is not a valid name.'%args.name)
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    for task in tasks:
        intensities = read_intensities(task, DATA_DIR)
        mask = read_mask(task, DATA_DIR)
        imgs = read_images(task, range(1, 97), intensities, mask, DATA_DIR)
        directions = read_directions(task, DATA_DIR)
        if args.improve:
            lambert_improved(task, imgs, directions, mask)
        else:
            lambert(task, imgs, directions)
        print('%s saved in %s'%(task, RESULT_DIR))
