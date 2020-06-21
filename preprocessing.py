import pandas as pd
import sys
import os
import os.path
import numpy as np
import re
import string
import scipy.io
import skimage
import skimage.measure
from skimage import io

# Set basic paths
data_path = os.getcwd() + '/Data/'
save_path = data_path

# Set image rows and cols
image_rows = 512
image_cols = 512

# To get data from each class folder
with open(os.getcwd() + '/foldernamelist.txt', 'r') as f:
    class = f.read().split('\n')
    
class_list = []

for i in range(len(class)):
    class_path.append(class[i] + str('/'))

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    
def label_writer(number, raw_img_sort):
    # Convert list of sorted raw image into dataframe
    raw_df = pd.DataFrame(raw_img_sort, columns=['Image'])
    
    # Add labels
    raw_df['Class'] = class[number] 
    raw_df.loc[(raw_df['Class']=='Normal'), "Label"] = 'Normal' 
    raw_df.loc[(raw_df['Class']!='Normal'), "Label"] = 'Abormal'
        
    if number == 0:
        label_df = raw_df
    else:
        label_df = pd.read_csv(save_path + class[number-1] + '.csv')
        del label_df['Unnamed: 0']
        label_df = pd.merge(label_df, raw_df, how='outer')
    
    # Remove names including '.csv' and '.npy'
    label_df = label_df[~label_df.Image.str.contains('|'.join(['.csv', '.npy']))]
        
    return label_df

def create_train_data(number):
    # Get raw data path
    img_path = os.path.join(data_path, class_list[number])
    print(img_path)

    img_files = os.listdir(img_path)
    print("We have " + str(len(img_files)) + " images on " + class[number] + " folder.")
    
    # Sort images by number
    raw_img_sort = sorted_alphanumeric(img_files)
    
    # Save image label
    label_df = label_writer(number, raw_img_sort)
    label_df.to_csv(save_path + class[number] + '.csv')
    
    print('Saving image label done.')
    
    reshape_img_s = np.ndarray((0, image_rows, image_cols, 3), dtype=np.uint8)
    reshape_img = np.ndarray((1, image_rows, image_cols, 3), dtype=np.uint8)

    Totals = np.ndarray((0, image_rows, image_cols, 1), dtype=np.uint8)

    print('Creating training images.')
        
    for img_name in raw_img_sort:
        # Get single image inside sorted img folder
        img_name_X = img_name
        
        # Path of single image selected
        img_path_X = os.path.join(img_path, img_name_X)
        
        # Read image
        img_size = io.imread(img_path_X, pilmode="RGB")
        
        if img_size.shape[1] > 200:
            reshape_img[0,:,:,:] = io.imread(img_path_X, pilmode="RGB")           
            reshape_img_s = np.concatenate((reshape_img[:],reshape_img_s[:]))

            print('X_image:', img_name_X)
            print('X_image (reshape):', reshape_img_s.shape)
            
            np.save(save_path + class[number] + '.npy', reshape_img_s)   
    
    print(str(number) + ' stage done.')
    
    return reshape_img_s
    
for i in range(len(class_list)):
    reshaper_img = create_train_data(i)
