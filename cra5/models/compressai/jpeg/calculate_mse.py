import  numpy as np
import json
from glob import glob
import cv2
import os
root = "/mnt/petrelfs/hantao.dispatch/NWP/comp_era5"
target = "jpeg2000_bit16_dB40"

def main(targetfolder):
    count=0
    mse = 0
    files = glob(os.path.join(root, 'numpy',"*.npy"))
    with open(os.path.join(root,'shift_scale.json' ), mode='r') as f:
        shift_scale = json.load(f)
    for idx, file in enumerate(files):
        gt = np.load(file)
        pre_file = file.replace(".j2k.npy", '.png').replace('numpy', target)
        pred = cv2.imread(pre_file,cv2.IMREAD_GRAYSCALE)
        file_index = os.path.split(pre_file)[1].split('.')[0]
        lower, upper = shift_scale[file_index]
        if pred is  not None:
            pred = pred/255.
            pred = pred*upper+lower
            # import pdb
            # pdb.set_trace()


            mse = mse*count + np.mean(np.square(gt-pred))
            count +=1
            mse/=count

        if count%1000==0:
            print(mse)
    print(mse)


if __name__ == '__main__':

    main(target)