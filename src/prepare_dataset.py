import sys
import glob
import os
import skimage.io
import skimage.filters
import skimage.color
import skimage.transform
import random
import pandas

def rgb2bin(img):
    bw = skimage.color.rgb2gray(img)
    threshold = skimage.filters.threshold_otsu(bw)
    return bw > threshold

if __name__ == '__main__':

    random.seed(42)
    if len(sys.argv) < 6:
        print('Usage: python3 prepare_dataset.py image_directory output_directory row_size col_size metadata')
        sys.exit()

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    row_size = int(sys.argv[3])
    col_size = int(sys.argv[4])
    metadata = pandas.read_csv(sys.argv[5])
    image_files = glob.glob(os.path.join(input_dir, '*.png'))
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    for digit in pandas.unique(metadata['digit']):
        os.makedirs(os.path.join(output_dir, 'train', str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'validation', str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', str(digit)), exist_ok=True)

    for index, row in metadata.iterrows():
        img_fname = os.path.join(input_dir, row['filename'])
        print(row['filename'])
        digit = str(row['digit'])
        img = skimage.io.imread(img_fname)
        img = rgb2bin(img)
        img = skimage.transform.resize(img, (row_size, col_size))
        img = skimage.img_as_float32(img)
        rand_num =  random.random()
        if rand_num < 0.15:
            skimage.io.imsave(os.path.join(output_dir, 'test', digit, os.path.basename(img_fname)), img)
        elif rand_num < 0.30:
            skimage.io.imsave(os.path.join(output_dir, 'validation', digit, os.path.basename(img_fname)), img)
        else:    
            skimage.io.imsave(os.path.join(output_dir, 'train', digit, os.path.basename(img_fname)), img)

