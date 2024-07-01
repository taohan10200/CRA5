# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import math
from PIL import Image
import json
import torch
import torch.nn.functional as F
import random
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr
from torch.utils.data import Dataset
import time
from multiprocessing import shared_memory,Pool
import multiprocessing
import copy
import queue
import threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from mmengine.dataset import Compose
from nwp.registry import DATASETS
from  .s3_client import s3_client
import tempfile
# from petrel_client.client import Client
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
variable_keys = [
    'Z_GDS0_ISBL',
    'V_GDS0_ISBL',
    'T_GDS0_ISBL',
    'Q_GDS0_ISBL',
    'U_GDS0_ISBL',
    'W_GDS0_ISBL',
    'VO_GDS0_ISBL',
    'D_GDS0_ISBL',
    'R_GDS0_ISBL',
    'O3_GDS0_ISBL',
    'CLWC_GDS0_ISBL',
    'CIWC_GDS0_ISBL',
    'CSWC_GDS0_ISBL',
    'CC_GDS0_ISBL',
    'CRWC_GDS0_ISBL',
    'PV_GDS0_ISBL',
    'initial_time0_hours',
    'initial_time0_encoded',
    'g0_lat_2',
    'g0_lon_3',
    'lv_ISBL1',
    'initial_time0']

variable_to_vname = {
    'Z_GDS0_ISBL': 'geopotential',
    'U_GDS0_ISBL': 'u_component_of_wind',
    'PV_GDS0_ISBL': 'potential_vorticity',
    'T_GDS0_ISBL': 'temperature',
    'V_GDS0_ISBL': 'v_component_of_wind',
    'Q_GDS0_ISBL': 'specific humidity',
    'W_GDS0_ISBL': 'vertical_velocity',
    'VO_GDS0_ISBL': 'vorticity',
    'O3_GDS0_ISBL': 'ozone_mass_mixing_ratio',
    'D_GDS0_ISBL': 'divergence',
    'R_GDS0_ISBL': 'relative_humidity',
    'CLWC_GDS0_ISBL': 'specific_cloud_liquid_water_conten',
    'CIWC_GDS0_ISBL': 'specific_cloud_ice_water_content',
    'CC_GDS0_ISBL': 'fraction_of_cloud_cover',
    'CRWC_GDS0_ISBL': 'specific_rain_water_content',
    'CSWC_GDS0_ISBL': 'specific_snow_water_content',
}

total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
 775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
 350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
 70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
 1.]

def bytes_to_string(n):
    u = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1024:
        n /= 1024.0
        i += 1
    return "%g%s" % (int(n * 10 + 0.5) / 10.0, u[i])
@DATASETS.register_module()
class era5_base_nc(Dataset):

    def __init__(self,
                 pipeline,
                 year,
                 data_prefix,
                 s3_bucket,
                 bucket_name,
                 test_mode,
                 format,
                 vname,
                 pressure_level=None,
                 is_norm=False,
                 time_interval='6H',
                 mae_pretrain=False,
                 sequence_cfg=None,
                 test_val=False,
                 num_samples=None,
                 multi_scale=True,
                 use_time_fea = False,
                 flip=True,
                 ori_size = (128, 256),
                 crop_size=(128, 256),
                 downsample_rate=1,
                 scale_factor=(0.5,1/0.5),
                 norm_type='overall',
                 iter = None,
                 finetune_interval = None,
                 num_workers = 1,
                 finetune_window = 0
                 ):

        super(era5_base_nc, self).__init__()
        self.pipeline = Compose(pipeline)
        self.s3_bucket = s3_bucket
        self.vnames = vname
        self.format=format
        self.crop_size = crop_size
        self.ori_size = ori_size
        self.multi_scale = multi_scale
        self.flip = flip
        self.scale_factor =scale_factor
        self.norm_type=norm_type
        self.is_norm = is_norm
        self.sequence_cfg=sequence_cfg
        self.test_val=test_val
        self.mae_pretrain = mae_pretrain
        self.use_time_fea = use_time_fea
        self.shift_scale_json={}
        self.iter = iter
        self.finetune_interval = finetune_interval
        self.num_workers = num_workers
        self.finetune_window = finetune_window
        self.test_mode = test_mode
        self.bucket_name = bucket_name
        self.timestamps = pd.date_range(year['start'],year['end']+'-23',freq=time_interval) #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.time_interval = int(time_interval.strip('H'))
        # 转换为 np.datetime64 类型，保留分钟精度

        if num_samples:
            self.timestamps = self.timestamps[num_samples:]
            
        self.timestamps_np = np.array(self.timestamps).astype('datetime64[s]')

        self.file_list= [os.path.join( str(time_stamp)[:4], str(time_stamp).replace('T','/'))
                     for time_stamp in self.timestamps_np]
        
        if self.s3_bucket is not None:
            self.data_root = data_prefix
            self.s3_client = s3_client(**self.s3_bucket)
        else:
            self.s3_client = None
            self.data_root = data_prefix
            print('s3_client is not available because you did not specify a right bucket name')
        if pressure_level is not None:
            self.pressure_level =pressure_level
            self.level_mapping =  [total_levels.index(val) for val in pressure_level if val in total_levels ]
        else:
            self.pressure_level = total_levels
            self.level_mapping = [idx for (idx, val) in enumerate(total_levels)]

        self.mean_std = self.get_mean_std()
        self.hour_onehot= np.eye(24,24).astype(np.float32)
        self.month_onehot= np.eye(12,12).astype(np.float32)
        

        self.tmp_list = []
        self.thread_start=False
        self.channels_to_vname = {}
        ch_idx = 0
        for v in self.vnames.get('pressure'):
            for level in self.pressure_level:
                self.channels_to_vname.update({ch_idx: v+'_'+str(int(level)) })
                ch_idx += 1
        for v in self.vnames.get('single'):
            self.channels_to_vname.update({ch_idx: v })
            ch_idx += 1


        self._metainfo = {'sequence_cfg':self.sequence_cfg,
                          'channels_to_vname':self.channels_to_vname,
                          'clim_mean':None,'mean':self.mean_std['mean'], 'std':self.mean_std['std']}


        #=========init multi processing =====
        self.data_element_num = len(self.vnames.get('single')) + \
                                len(self.vnames.get('pressure')) * len(self.pressure_level)
        self.index_dict1 = {}
        self.index_dict2 = {}

        i = 0
        for vname in self.vnames.get('pressure'):
            for height in self.pressure_level:
                self.index_dict1[(vname, height)] = i
                i += 1
        for vname in self.vnames.get('single'):
            self.index_dict1[(vname, 0)] = i
            i += 1

        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()

        self.index_queue.cancel_join_thread()
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}


        self.compound_data_queue_num = 20

        self.lock = multiprocessing.Lock()

        self.a = np.zeros((len( self.channels_to_vname.keys()),
                           self.ori_size[0], self.ori_size[1]), dtype=np.float32)

        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.a.nbytes)
            self.sharedmemory_list.append(shm)

        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))



        self._workers = []

        for _ in range(len(self.channels_to_vname.keys())//2+1): #len(self.channels_to_vname.keys())//2
            w = multiprocessing.Process(
                target=self.load_data_process)
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._workers.append(w)
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)

    def __len__(self):
        samples_num = len(self.file_list)
        gt_idx = self.sequence_cfg.get('gt')
        input_idx = self.sequence_cfg.get('input')
        samples_num -= (math.ceil(gt_idx[-1]/self.time_interval) + math.ceil(input_idx[-1]/self.time_interval)) 
        return samples_num

    def get_mean_std(self):
        with open('./nwp/datasets/mean_std.json',mode='r') as f:
            mean_std = json.load(f)
            f.close()
        with open('./nwp/datasets/mean_std_single.json',mode='r') as f:
            mean_std_single = json.load(f)
            f.close()

        mean_list, std_list = [],[]
        for  vname in self.vnames.get('pressure'):
            if self.norm_type == 'overall':
                mean_list += [mean_std['mean'][vname+'_overall']]*len(self.pressure_level)
                std_list += [mean_std['std'][vname+'_overall']]*len(self.pressure_level)
                
            elif self.norm_type == 'channel':
                mean_list += [mean_std['mean'][vname][idx] for idx in self.level_mapping]
                std_list += [mean_std['std'][vname][idx] for idx in self.level_mapping]
            else:
                raise ValueError("norm_type must be the channel or overall !!!" )

        for vname in self.vnames.get('single'):
            mean_list.append(mean_std_single['mean'][vname])
            std_list.append(mean_std_single['std'][vname])
        
        return dict(mean=np.array(mean_list,dtype=np.float32), std=np.array(std_list, dtype=np.float32))

    def normalization(self, data):
        data -= self.mean_std['mean'][:,np.newaxis,np.newaxis]
        data /= self.mean_std['std'][:,np.newaxis,np.newaxis]
        return data

    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid, idx) in recorder_dict:
                recorder_dict[(job_pid, idx)][(vname, height)] = 1
            else:
                recorder_dict[(job_pid, idx)] = {(vname, height): 1}
            if len(recorder_dict[(job_pid, idx)]) == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))

    def load_data_process(self):
        while True:
            job_pid, idx, vname, height,bucket_name = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()

            if vname in self.vnames.get('single'):
                file = os.path.join('single',idx)
                if bucket_name is None:
                    url = f"{self.data_root}{file}{vname}.npy"
                else:
                    url = f"{file}{vname}.npy"
            elif vname in  self.vnames.get('pressure'):
                file =  idx
                if bucket_name is None:
                    url = f"{self.data_root}{file}{vname}-{height}.npy"
                else:
                    url = f"{file}{vname}-{height}.npy"
            b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
           
            if self.s3_client is not None:
                unit_data = self.s3_client.read_npy_from_BytesIO(url,bucket_name)

            else:
                unit_data =np.load(url)

            if unit_data.shape[-1]!=self.a.shape[-1]:
                unit_data = np.resize(unit_data, self.a.shape[-2:])

            b[self.index_dict1[(vname, height)]] = unit_data
          
            self.unit_data_queue.put((job_pid, idx, vname, height))

    def get_data(self, idxes, bucket_name=None, return_norm=True):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)


            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err

        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)

        return_data = []
        for idx in idxes:
            for vname in self.vnames.get('pressure'):
                for height in self.pressure_level:
                    self.index_queue.put((job_pid, idx, vname, height,bucket_name))

            for vname in self.vnames.get('single'):
                self.index_queue.put((job_pid, idx, vname, 0, bucket_name))

            idx = self.compound_data_queue_dict[job_pid].get()

            data_tmp = copy.deepcopy(b)
            if not self.is_norm and return_norm:
                data_tmp = self.normalization(data_tmp)
            return_data.append(data_tmp)

        return return_data

    def get_data_nc(self, idxes, read='all'):
        return_data = []

        for idx in idxes:
            url = f"{self.data_root}/{idx[:-1]}.grib"
            tmp1 = tempfile.NamedTemporaryFile(prefix='/nvme/')
            pressure_data = self.s3_client.read_from_tmpfile(url,tmp1.name)
            
            tmp2 = tempfile.NamedTemporaryFile(prefix='/nvme/')
            url = f"{self.data_root}/single/{idx[:-1]}.grib"
            single_data = self.s3_client.read_from_tmpfile(url,tmp2.name)

            hours = pressure_data.time.data
            steps = pressure_data.step.data
            # import pdb
            # pdb.set_trace()
            one_item = []
            for step in steps:
                each_step= []
                pressure_per_step = pressure_data.sel(step=step)
                single_per_step   = single_data.sel(step=step)
                
                for vname in self.vnames.get('pressure'):
                    D = pressure_per_step[vname].data
                    Pha_levels = list(pressure_per_step.isobaricInhPa.data)
                    if D.ndim==2 and len(Pha_levels)==1:
                        each_step.append(D[None])
                    else:
                        level_mapping =  [Pha_levels.index(val) for val in self.pressure_level if val in Pha_levels]
                        for level in level_mapping:
                            each_step.append(D[level][None])

                for vname in self.vnames.get('single'):
                    D = single_per_step[vname].data
                    each_step.append(D[None])


                each_step = np.concatenate(copy.deepcopy(each_step),0)
                if not self.is_norm:
                    data_tmp = self.normalization(each_step)
                # self.save_as_jpeg2000(data_tmp,idx)
                one_item.append(data_tmp)
                if read == 'first_step':
                    break
                
            return_data.append(one_item)
            # import pdb
            # pdb.set_trace() 
            tmp1.close()
            tmp2.close()
        return return_data
    def get_prediction_from_nc(self, idxes, bucket_name=None):
        return_data = []
        for url in idxes:
            pressure_data = self.s3_client.read_nc_from_BytesIO(url, bucket=bucket_name)
            
            url = f"single/{url}"
            single_data = self.s3_client.read_nc_from_BytesIO(url,  bucket=bucket_name)

            hours = pressure_data.time.data

            each_step= []
            # pressure_per_step = pressure_data.sel(step=step)
            # single_per_step   = single_data.sel(step=step)
            
            for vname in self.vnames.get('pressure'):
                D = pressure_data[vname].data
                Pha_levels = list(pressure_data.isobaricInhPa.data)
                if D.ndim==2 and len(Pha_levels)==1:
                    each_step.append(D[None])
                else: 
                    level_mapping =  [Pha_levels.index(val) for val in self.pressure_level if val in Pha_levels]
                    for level in level_mapping:
                        each_step.append(D[0][level][None])

            for vname in self.vnames.get('single'):
                D = single_data[vname].data

                each_step.append(D)


            each_step = np.concatenate(copy.deepcopy(each_step),0)

            if not self.is_norm:
                each_step = self.normalization(each_step)
            # self.save_as_jpeg2000(data_tmp,idx)

            return_data.append(each_step)

        return return_data
    
    def get_prediction(self, idxes, bucket_name=None):
        return_data = []
        for idx in idxes:
            each_step= []

            for vname in self.vnames.get('pressure'):
                for level in self.pressure_level:
                    url = idx+ f'{vname}-{level}.npy'
                    D = self.s3_client.read_npy_from_BytesIO(url, bucket_name)
                    if D.ndim==2:
                        each_step.append(D[None])

            for vname in self.vnames.get('single'):
                url = idx+ f'{vname}.npy'
                D = self.s3_client.read_npy_from_BytesIO(url,bucket_name)
                each_step.append(D[None])

            # import pdb
            # pdb.set_trace()
            each_step = np.concatenate(copy.deepcopy(each_step),0)
            # if not self.is_norm:
            #     each_step = self.normalization(each_step)
            # self.save_as_jpeg2000(data_tmp,idx)
            return_data.append(each_step)
            
        return return_data
    

    
    def __getitem__(self, index):

        in_timestamp = [self.timestamps[index]+timedelta(hours=i) for i in self.sequence_cfg.get('input')]
        gt_timestamp = [in_timestamp[-1] + timedelta(hours=i) for i in self.sequence_cfg.get('gt')]

        
        in_path = [os.path.join( str(stamp)[:4], str(stamp).replace(' ','/')+'-' )  for stamp in in_timestamp]
        gt_path = [os.path.join( str(stamp)[:4], str(stamp).replace(' ','/')+'-' )  for stamp in gt_timestamp]
        
        if self.test_mode=='ensemble': #self.s3_bucket.get('bucket_name') == 'nwp_initial_fileds' :

            prediction = self.get_data_nc(in_path, read='all')[0]
            input_queue =  prediction[0]
            # get the prediciotn of physical model, 2 is the interval to get the 12 hour slot data. 
            # index 2 is the first 12hour prediction  
            pred_queue = [prediction[i] for i in range(2, len(prediction), 2)][:len(gt_path)]

            # import pdb
            # pdb.set_trace()

            gt_queue = self.get_data_nc(gt_path, read='first_step') # if the read=initail, it will return the zero initial field.
            gt_queue = [i[0] for i in gt_queue]

        elif self.test_mode == 'HRES_25km': #self.s3_bucket.get('bucket_name')== 'nwp_predictions':
            prediction = self.get_data_nc(in_path, read='all')[0]
            input_queue =  prediction[0]
            # import pdb
            # pdb.set_trace()
            pred_queue = [prediction[i] for i in range(0, len(prediction), 1)][:len(gt_path)]
            gt_queue =  self.get_data(gt_path, bucket_name=self.bucket_name.get('gt_bucket')) #'era5_np_float32'
            
        elif self.test_mode == 'operational_9km': #self.s3_bucket.get('bucket_name')== 'nwp_predictions':
            prediction = self.get_data_nc(in_path, read='all')[0]
            print(in_path)
            input_queue =  prediction[0]
            pred_queue = [prediction[i] for i in range(0, len(prediction), 1)][:len(gt_path)]
            gt_queue =  self.get_data(gt_path, bucket_name='nwp_initial_fileds/analysis_MIR/np2001x4000/') #'era5_np_float32'   
        
        elif self.test_mode == 'aimodel_9km': #self.s3_bucket.get('bucket_name')== 'nwp_predictions' :
            pred_path = [f"{str(in_timestamp[0])[:4]}/{str(in_timestamp[0]).replace(' ','T')}/{str(pred_time).replace(' ','T')}.nc" for pred_time in gt_timestamp]

            pred_queue = self.get_prediction_from_nc(pred_path,  bucket_name='nwp_predictions/FengWu_GHR/')
            
            gt_queue =  self.get_data(gt_path, bucket_name='nwp_initial_fileds/analysis_MIR/np2001x4000/') #'era5_np_float32'
            input_queue =  [pred_queue[0]]
        elif self.test_mode == 'aimodel_9km_extreme': #self.s3_bucket.get('bucket_name')== 'nwp_predictions' :
            pred_path = [f"{str(in_timestamp[0])[:4]}/{str(in_timestamp[0]).replace(' ','T')}/{str(pred_time).replace(' ','T')}.nc" for pred_time in gt_timestamp]

            pred_queue = self.get_prediction_from_nc(pred_path,  bucket_name='nwp_predictions/FengWu_GHR_extreme/')
           
            gt_queue =  self.get_data(gt_path, bucket_name='nwp_initial_fileds/analysis_MIR/np2001x4000/') #'era5_np_float32'
            input_queue =  [pred_queue[0]]
        elif self.test_mode == 'aimodel_9km_to_25km': 
            pred_path = [f"{str(in_timestamp[0])[:4]}/{str(in_timestamp[0]).replace(' ','T')}/{str(pred_time).replace(' ','T')}.nc" for pred_time in gt_timestamp]

            pred_queue = self.get_prediction_from_nc(pred_path,  bucket_name='nwp_predictions/FengWu_GHR/')
            
            gt_queue =  self.get_data(gt_path, bucket_name='nwp_initial_fileds/analysis_MIR/np721x1440/') #'era5_np_float32'
            input_queue =  [pred_queue[0]]

            pred_queue = [F.interpolate(torch.from_numpy(i)[None], size=gt_queue[0].shape[-2:]).squeeze().numpy() for i in pred_queue]

        elif self.test_mode == 'aimodel_25km': #self.s3_bucket.get('bucket_name')== 'nwp_predictions' :
            # pred_path = [os.path.join( str(in_timestamp[0]).replace(' ','T')+'/'+ str(12*idx)+'/') 
                        #  for idx, _  in enumerate(gt_timestamp,1)]
            pred_path = [f"{str(in_timestamp[0])[:4]}/{str(in_timestamp[0]).replace(' ','T')}/{str(pred_time).replace(' ','T')}.nc" for pred_time in gt_timestamp]

            pred_queue = self.get_prediction_from_nc(pred_path,  bucket_name=self.bucket_name.get('pred_bucket'))
            
            # pred_queue = self.get_prediction(pred_path, bucket_name=self.bucket_name.get('pred_bucket'))
            
            gt_queue =  self.get_data(gt_path, bucket_name=self.bucket_name.get('gt_bucket')) #'era5_np_float32'
            pred_queue
            input_queue =  [pred_queue[0]]

        else:
            input_queue = self.get_data(in_path)
            gt_queue = self.get_data(gt_path)

        input, gt_label = np.concatenate(copy.deepcopy(input_queue),0), \
                          np.array(copy.deepcopy(gt_queue),dtype=np.float32)
                          
        pred_label  =  np.array(copy.deepcopy(pred_queue),dtype=np.float32)   
        
        del input_queue,gt_queue,pred_queue
        
        assert isinstance( self.timestamps_np[0], np.datetime64)
        
        data_dict = {'input':input,
                'gt_label':gt_label,
                'pred_label':pred_label,
                'in_sample_idx':in_timestamp,
                'gt_sample_idx':gt_timestamp,
                'in_time_stamp':np.array(in_timestamp).astype('datetime64[s]'),
                'gt_time_stamp':np.array(gt_timestamp).astype('datetime64[s]'),
                'in_ori_shape':input[0].shape,
                'gt_ori_shape':input[0].shape,
                }
        if self.test_val == 'test':
            climate_path =  [os.path.join('climate_mean_day/1993-2016/', str(stamp)[5:10]+'/')  for stamp in gt_timestamp]
            climate_queue = self.get_data(climate_path, bucket_name=self.bucket_name['climate_bucket'], return_norm=False)
            climate_mean = np.array(copy.deepcopy(climate_queue),dtype=np.float32)
            data_dict.update({'climate_mean':climate_mean})

            del climate_queue
        
        data = self.pipeline(data_dict)
        return  data
    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def check_img(self, image,divisor ):
        h, w = image.shape[:2]
        if h % divisor != 0:
            real_h = h+(divisor - h % divisor)
        else:
            real_h = h
        if w % divisor != 0:
            real_w = w+(divisor - w % divisor)
        else:
            real_w = 0
        image = self.pad_image(image, size=(real_h, real_w), h=h,w=w, padvalue= (0.0, 0.0, 0.0))
        return image



    def crop_patch(self,image, gt_label):

        th, tw = self.crop_size[0], self.crop_size[1]

        # image = self.pad_image(image, h, w, (th, tw), (0.0, 0.0, 0.0))

        c,h, w = image.shape
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)
        image = image[:,y:y + th, x:x + tw]
        gt_label = gt_label[:,y:y + th, x:x + tw]

        return image, gt_label

    def crop_then_scale(self,image, points, scale_factor, crop_size):
        # th, tw = self.crop_size[0], self.crop_size[1]

        th, tw = int(round(crop_size[0]/scale_factor)), int(round(crop_size[1]/scale_factor))
        h, w = image.shape[:2]

        image = self.pad_image(image, h, w, (th, tw), (0.0, 0.0, 0.0))

        h, w = image.shape[:2]
        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)
        image = image[y:y + th, x:x + tw]
        image = cv2.resize(image, (crop_size[1], crop_size[0]),
                           interpolation=cv2.INTER_LINEAR)  # width, height
        return image, points

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label
    def multi_scale_aug(self, image, label=None,
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))

if __name__ == '__main__':
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    bucket_name='era5_np128x256'
    local_root = '/mnt/petrelfs/hantao.dispatch/NWP/era5_local/era5_np128x256/'  #'/nvme/hantao/era5/era5_np128x256/'
    data_prefix='era5_np128x256'
    format='.npy'
    vname=dict(
        pressure=['z','q', 'u', 'v', 't','w'],
        single=['v10','u10','t2m','sp','msl']#'z','lsm'],
        # constant=['z','lsm'] #,'slt'
    )


