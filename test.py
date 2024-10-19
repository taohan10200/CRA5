from cra5.api.era5_downloader import era5_downloader
# ERA5_data = era5_downloader('./cra5/api/era5_config.py') #specify the dataset config for what we want to download
# data = ERA5_data.get_form_timestamp(time_stamp="2024-06-01T00:00:00",
#                                     local_root='./data/ERA5')
import time
from cra5.api import cra5_api
import numpy as np
cra5_API = cra5_api()


encoding_time = []
decoding_time = []
for i in range(100):
	####=======================compression functions=====================
	# Return a continuous latent y for ERA5 data at 2024-06-01T00:00:00

	y = cra5_API.encode_to_latent(time_stamp="2024-06-01T00:00:00") 

	# Return the the arithmetic coded binary stream of y 
	bin_stream = cra5_API.latent_to_bin(y=y)  


	# Or if you want to directly compress and save the binary stream to a folder
	output = cra5_API.encode_era5_as_bin(time_stamp="2024-06-01T00:00:00", save_root='./data/CRA5')  
	encoding_time.append(output['encoding_time'])

	####=======================decompression functions=====================
	# Starting from the bin_stream, you can decode the binary file to the quantized latent.
	y_hat = cra5_API.bin_to_latent(bin_path="./data/CRA5/2024/2024-06-01T00:00:00.bin")  # Decoding from binary can only get the quantized latent.

	# Return the normalized cra5 data
	normlized_x_hat = cra5_API.latent_to_reconstruction(y_hat=y_hat) 

	#or 
 
	# If you have saveed  or downloaded the binary file, then you can directly restore the binary file into reconstruction.
	output = cra5_API.decode_from_bin("2024-06-01T00:00:00", return_format='normalized') # Return the normalized cra5 data
	normlized_x_hat = output['x_hat']


	output = cra5_API.decode_from_bin("2024-06-01T00:00:00", return_format='de_normalized') # Return the de-normalized cra5 data
	x_hat = output['x_hat']
	decoding_time.append(output['decoding_time'])
	print(f'step_{i}')
	
#  # Show some channels of the latent
# 	cra5_API.show_latent(
# 		latent=y_hat.squeeze(0).cpu().numpy(), 
# 		time_stamp="2024-06-01T00:00:00", 
# 		show_channels=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
# 		save_path = './data/CRA5_vis')

# 	# show some variables for the constructed data
# 	cra5_API.show_image(
# 		reconstruct_data=x_hat.cpu().numpy(), 
# 		time_stamp="2024-06-01T00:00:00", 
# 		show_variables=['z_500', 'q_500', 'u_500', 'v_500', 't_500', 'w_500'],
# 		save_path = './data/CRA5_vis')
print(f"The average encoding time is: {np.array(encoding_time).mean()}s variance: {np.array(encoding_time).std()} s")
print(f"The average deconding time is: {np.array(decoding_time).mean()}s variance: {np.array(decoding_time).std()} s")