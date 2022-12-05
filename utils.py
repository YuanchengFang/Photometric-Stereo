import cv2
import numpy as np
import os

def read_mask(task, datadir='./pmsData') -> np.ndarray:
    filename = os.path.join(datadir, f'{task}PNG/mask.png')
    mask = cv2.imread(filename)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask / 255.0
    return mask

def read_intensities(task, datadir='./pmsData') -> np.ndarray:
    filename = os.path.join(datadir, f'{task}PNG/light_intensities.txt')
    with open(filename, 'r') as f:
        intensities = f.readlines()
    intensities = [[float(i) for i in line.strip().split(' ')] for line in intensities]
    return np.array(intensities)

def read_directions(task, datadir='./pmsData') -> np.ndarray:
    filename = os.path.join(datadir, f'{task}PNG/light_directions.txt')
    with open(filename, 'r') as f:
        directions = f.readlines()
    directions = [[float(i) for i in line.strip().split(' ')] for line in directions]
    return np.array(directions)


def read_images(task, ids, intensities, mask, datadir='./pmsData') -> np.ndarray:
    """Read images into (N, H, W) shape array, where N equals to len(ids)

    Parameters
    -------------
    task: str
        task name, in ['bear', 'buddha', 'cat', 'pot']
    ids: List[int]
        the ids of needed images
    intensities:
        light intensities with shape (N, 3)
    mask:
        mask image array with shape (H, W)
    """
    images = []
    for i in ids:
        # READ image
        filename = os.path.join(datadir, f'{task}PNG/{i:03d}.png')
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # NORMALIZE wtih intensities
        img = img / intensities[i-1, :] / (2**16-1)
        
        # BGR to GRAY
        img = 0.114*img[:,:,0] + 0.587*img[:,:,1] + 0.299*img[:,:,2]
        
        # MASK
        img = img * mask 
        images.append(img)
    images = np.stack(images, axis=0)
    return images
    