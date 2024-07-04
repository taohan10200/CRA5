import sys
import os
import json
import torch
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
from era5_downloader import era5_downloader
if importlib.util.find_spec("mmengine") is not None and sys.version_info >= (2, 0):
    from mmengine import Config, DictAction
else:
    from mmcv import Config, DictAction
import numpy as np
import xarray as xr
from pathlib import Path
from .utils import (filesize, write_uints, write_bytes, read_uints, read_bytes)
from cra5.models.compressai.zoo import vaeformer_pretrained

current_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_path)
work_directory = os.getcwd()
print(directory_path, work_directory)

class cra5_api():
    def __init__(self, 
                 config=f'{directory_path}/cra5_268v_config.py',
                 local_root=f'{work_directory}/data',
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 ):
        self.device = device 
        print(f'The serving device is {self.device}')
        self.cfg = Config.fromfile(config)
        self.era5 = era5_downloader(f'{directory_path}/era5_config.py')
        self.level_mapping = [self.cfg.total_levels.index(val) \
            for val in self.cfg.pressure_level if val in self.cfg.total_levels ]
        self.mean, self.std = self.get_mean_std()
        self.mean = torch.from_numpy(self.mean[:,np.newaxis,np.newaxis]).to(device)
        self.std = torch.from_numpy(self.std[:,np.newaxis,np.newaxis]).to(device)
        
        self.channels_to_vname, self.vname_to_channels = self.channel_vname_mapping()
        print(self.channels_to_vname)
        self.local_root = local_root
        self.net = vaeformer_pretrained(quality=268, pretrained=True).eval().to(device)
        
    def download_era5_data(self, 
                           time_stamp:str, 
                           save_root=None,
                           data_formate="nc"):
        save_root = save_root or self.local_root
        data = self.era5.get_form_timestamp(
                                    time_stamp=time_stamp,
                                    local_root=save_root,
                                    )
        
    def encoder_era5(self,
                     time_stamp:str,
                     save_root=None, 
                     ):
        
        save_root = save_root or self.local_root
        # self.download_era5_data(time_stamp)
        data = self.read_data_from_grib(time_stamp)
        data = torch.from_numpy(data).to(self.device)
        x = self.normalization(data).unsqueeze(0)

        with torch.no_grad():
            st = time.time()
            output = self.net.compress(x) 
            print(f'The encoding time is {time.time()-st} s')
        print(output["z_shape"])
        
        year = time_stamp.split('-')[0]
        file_url=f'{save_root}/cra5/{year}/{time_stamp}.bin'
        print(os.path.dirname(file_url))
    
        os.makedirs(os.path.dirname(file_url), exist_ok=True)    
        with Path(file_url).open("wb") as f:
            out_strings = output["strings"]
            shape = output["z_shape"]
            bytes_cnt = 0
            bytes_cnt = write_uints(f, (shape[0], shape[1], len(out_strings)))
        
            for s in out_strings:
                print(len(s))
                bytes_cnt += write_uints(f, (len(s[0]),))
                bytes_cnt += write_bytes(f, s[0])
        
    def decode_from_bin(self, 
                        time_stamp, 
                        return_format='de_normlized',
                        ):

        bin_path = f'{self.local_root}/cra5/{time_stamp[:4]}/{time_stamp}.bin'
        dec_start = time.time()
        with Path(bin_path).open("rb") as f:
            lstrings = []
            shape = read_uints(f, 2)

            n_strings = read_uints(f, 1)[0]
            # print(shape, n_strings)
            for _ in range(n_strings):
                s = read_bytes(f, read_uints(f, 1)[0])
                lstrings.append([s])
                
            with torch.no_grad():    
                print(f"Decoded in {time.time() - dec_start:.2f}s")
                if return_format=='latent':
                    output =  self.net.decompress(lstrings, shape, return_format='latent')
                    return output
                else:
                    output =  self.net.decompress(lstrings, shape)
                
            if return_format =='normlized':
                return output['x_hat']
           
            elif return_format =='de_normlized':
                x_hat = self.de_normalization(output['x_hat'].squeeze(0))
                print(f"Decoded in {time.time() - dec_start:.2f}s")
                                    
                return x_hat
    
            
    def read_data_from_grib(self,
                            time_stamp:str, 
                            ):
        one_step= []
        pressure_file = f'{self.local_root}/ERA5/{time_stamp[:4]}/{time_stamp}_pressure.nc'
        single_file = f'{self.local_root}/ERA5/{time_stamp[:4]}/{time_stamp}_single.nc'

        pressure_data = xr.open_dataset(pressure_file, 
                                        engine='netcdf4',
                                        )
        single_data = xr.open_dataset(single_file, 
                                        engine='netcdf4',
                                        )
                                            
        for vname in self.cfg.vnames.get('pressure'):
            D = pressure_data[vname].data
            Pha_levels = list(pressure_data.level.data)

            level_mapping =  [Pha_levels.index(val) for val in self.cfg.pressure_level if val in Pha_levels]
            
            for level in level_mapping:
                one_step.append(D[0][level][None])
                
        for vname in self.cfg.vnames.get('single'):
            D = single_data[vname].data
            if vname == 'tp':
                D = D * 1000
            one_step.append(D)

        one_step = np.concatenate(one_step,0)

        return one_step         
        
    def channel_vname_mapping(self):
        channels_to_vname={}
        vname_to_channels={}
        ch_idx = 0
        for v in self.cfg.vnames.get('pressure'):
            for level in self.cfg.pressure_level:
                channels_to_vname.update({ch_idx: v+'_'+str(int(level)) })
                vname_to_channels.update({v+'_'+str(int(level)): ch_idx })
                ch_idx += 1
        for v in self.cfg.vnames.get('single'):
            channels_to_vname.update({ch_idx: v })
            vname_to_channels.update({v: ch_idx})
            ch_idx += 1
        return channels_to_vname, vname_to_channels
    
    def get_mean_std(self):
        with open(f'{directory_path}/mean_std.json',mode='r') as f:
            mean_std = json.load(f)
            f.close()
        with open(f'{directory_path}/mean_std_single.json',mode='r') as f:
            mean_std_single = json.load(f)
            f.close()

        mean_list, std_list = [],[]

        for  vname in self.cfg.vnames.get('pressure'):
            mean_list += [mean_std['mean'][vname][idx] for idx in self.level_mapping]
            std_list += [mean_std['std'][vname][idx] for idx in self.level_mapping]

        for vname in self.cfg.vnames.get('single'):
            mean_list.append(mean_std_single['mean'][vname])
            std_list.append(mean_std_single['std'][vname])
            
        return np.array(mean_list,dtype=np.float32), np.array(std_list, dtype=np.float32)   
        

    def normalization(self, data):
        data = (data - self.mean)/self.std
        return data
    
    def de_normalization(self, data):
        data *=  self.std
        data +=  self.mean
        return data
    
    def show_image(self,
                   reconstruct_data,  
                   time_stamp,                           
                   show_variables:list=['z_500', 'q_500', 'u_500', 'v_500', 't_500', 'w_500'],
                   save_images=True,
                    ):
        input_data = self.read_data_from_grib(time_stamp)
        vis_data_list = []
        for vname in show_variables:
            data_ori = input_data[self.vname_to_channels[vname]]
            data_rec = reconstruct_data[self.vname_to_channels[vname]]
            diff = np.abs(data_ori - data_rec)
            vis_data_list.append([data_ori, data_rec, diff])
        # print(vis_data_list)

        fig, axs = plt.subplots(len(show_variables), 3, figsize=( 20, 3*len(show_variables)))

        for i, data in enumerate(vis_data_list):
            # original data
            im0 = axs[i, 0].imshow(data[0], cmap='jet')
            axs[i, 0].set_title(f'{show_variables[i]}_Original')
            fig.colorbar(im0, ax=axs[i, 0])
            
            #reconstrcted data
            axs[i, 1].imshow(data[1], cmap='jet')
            axs[i, 1].set_title(f'{show_variables[i]}_Reconstructed')
            fig.colorbar(im0, ax=axs[i, 1])
            
            #difference
            im2  = axs[i, 2].imshow(data[2], cmap='jet')
            axs[i, 2].set_title(f'{show_variables[i]}_Difference')
            fig.colorbar(im2, ax=axs[i, 2])
        plt.tight_layout()

        plt.show()
        fig_path = f'{self.local_root}/cra5_vis/{time_stamp[:4]}/{time_stamp}.png'
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        if save_images:
            plt.savefig(fig_path)
            
    def show_latent(self,
                   latent,  
                   time_stamp,                           
                   show_channels:list=[0, 10, 20, 30, 40, 50, 60, 70, 80],
                   save_images=True,
                    ):
        input_data = self.read_data_from_grib(time_stamp)


        fig, axs = plt.subplots(len(show_channels)//4, 4, figsize=( 24, 3*len(show_channels)//4))
        axs = axs.flatten()
        for i, cha_id in enumerate(show_channels):
            # original data
            im0 = axs[i].imshow(latent[cha_id], cmap='jet')
            axs[i].set_title(f'Channel_{cha_id}')
            fig.colorbar(im0, ax=axs[i])
        
        plt.tight_layout()
        plt.show()
        fig_path = f'{self.local_root}/cra5_vis/{time_stamp[:4]}/{time_stamp}_latent.png'
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        if save_images:
            plt.savefig(fig_path)