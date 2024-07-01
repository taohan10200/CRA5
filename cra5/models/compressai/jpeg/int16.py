import os
import pandas as pd
from PIL import Image
import numpy as np
import sys
import h5py

ext = sys.argv[2]
save_dict = {
	'j2k': 'JPEG2000',
	'png': 'PNG'	
}

all_files = os.listdir(sys.argv[1])

if not os.path.exists(ext):
	os.makedirs(ext)

for f in all_files:
	if '.mat' in f:
		print('load data ' + f)
		data = h5py.File(os.path.join(sys.argv[1], f),'r')
		img = np.array(data['rad'])
		# img *= 255.0/img.max()
		img = img.astype('uint16')

		save_path = os.path.join(ext, f.replace('.mat', ''))
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		len_img = len(img)

		for i in range(len_img):
			img8 = img[i]		
			img8 = Image.fromarray(img8)
			img_save_path = os.path.join(save_path, str(i) + '.' + ext)
			img8.save(img_save_path, save_dict[sys.argv[2]])
			print('saved image at ' + img_save_path)

		print()

		


