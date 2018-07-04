import sys
import glob
import os
import skimage.io
import skimage.filters
import skimage.color
import skimage.transform
import random
import pandas
import numpy as np

def get_bounding_box(img):
    row_nz, col_nz = np.nonzero(img)
    row_min = np.min(row_nz)
    row_max = np.max(row_nz)
    col_min = np.min(col_nz)
    col_max = np.max(col_nz)
    return img[row_min:row_max,col_min:col_max]

def rgb2bin(img):
    img = skimage.img_as_float32(img)
    bw = skimage.color.rgb2gray(img)
    threshold = skimage.filters.threshold_otsu(bw)
    mask = bw < threshold
    bw = np.zeros_like(bw)
    bw[mask] = 1.0
    bw = 1.0 - bw
    return bw

if __name__ == '__main__':

    random.seed(42)
    if len(sys.argv) < 6:
        print('Usage: python3 prepare_dataset.py image_directory output_directory row_size col_size metadata [no_split]')
        sys.exit()

    if len(sys.argv) == 7:
        no_split = True
    else:
        no_split = False
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    row_size = int(sys.argv[3])
    col_size = int(sys.argv[4])
    metadata = pandas.read_csv(sys.argv[5])
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    if not no_split:
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        validation_dir = os.path.join(output_dir, 'validation')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    for digit in pandas.unique(metadata['digit']):
        if not no_split:
            os.makedirs(os.path.join(train_dir, str(digit)), exist_ok=True)
            os.makedirs(os.path.join(test_dir, str(digit)), exist_ok=True)
            os.makedirs(os.path.join(validation_dir, str(digit)), exist_ok=True)
        else:
            os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    for index, row in metadata.iterrows():
        img_fname = os.path.join(input_dir, row['filename'])
        print(row['filename'])
        digit = str(row['digit'])
        img = skimage.io.imread(img_fname)
        img = rgb2bin(img)
        img = get_bounding_box(img)
        img = skimage.transform.resize(img, (max(img.shape), max(img.shape)),
                                        cval=0, mode='constant') 
        img = skimage.transform.resize(img, (row_size, col_size))
        if no_split:
            skimage.io.imsave(os.path.join(output_dir, digit, os.path.basename(img_fname)), img)
        else:
            rand_num =  random.random()
            if rand_num < 0.10:
                skimage.io.imsave(os.path.join(validation_dir, digit, os.path.basename(img_fname)), img)
            elif rand_num < 0.20:
                skimage.io.imsave(os.path.join(test_dir, digit, os.path.basename(img_fname)), img)
            else:    
                skimage.io.imsave(os.path.join(train_dir, digit, os.path.basename(img_fname)), img)

