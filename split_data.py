from imutils import paths
from tqdm import tqdm
import os
import numpy as np
import shutil

labels = os.listdir('mydata/train2')
# for label in labels:
#     os.mkdir("mydata/test/"+label)
for label in labels:
    imagePaths = os.listdir('mydata/train2/'+label)
    image_indexs = np.random.choice(imagePaths, size=len(imagePaths)//5, replace=False)
    for image_index in tqdm(image_indexs):
        try:
            shutil.move('mydata/train2/'+label+'/'+image_index, 'mydata/test/'+label)
        except:
            print(image_index)