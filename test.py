from cra5.api import cra5_api
cra5_API = cra5_api()
cra5_API.encoder_era5("2024-06-01T00:00:00")


cra5_data = cra5_API.decode_from_bin("2024-06-01T00:00:00")
cra5_API.show_image(
	reconstruct_data=cra5_data.cpu().numpy(), 
	time_stamp="2024-06-01T00:00:00", 
	show_variables=['z_500', 'q_500', 'u_500', 'v_500', 't_500', 'w_500'])

latent = cra5_API.decode_from_bin("2024-06-01T00:00:00", return_format='latent')
cra5_API.show_latent(
	latent=latent.squeeze(0).cpu().numpy(), 
	time_stamp="2024-06-01T00:00:00", 
	show_channels=[0, 10, 20, 30, 40, 50, 60, 70, 80,90,100,110, 120, 130, 140, 150])