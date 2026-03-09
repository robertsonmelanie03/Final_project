import numpy as np
import pdb
import random
import numpy as np
import subprocess
import urllib.request
import xarray as xr

def load_data(): 
    ds_bi = xr.open_dataset('bi_2025.nc')


    bi_var_name = 'burning_index_g'

    ds_pr = xr.open_dataset('pr_2025.nc')

    pr_var_name = 'precipitation_amount'


    ds_combined = xr.merge([ds_bi, ds_pr])
    #df = ds_combined.to_dataframe().reset_index()


    ds_sample = ds_combined.isel(day=slice(0,10), lat=slice(None, None, 10), lon=slice(None, None, 10))
    df_sample = ds_sample.to_dataframe().reset_index()


    df_sample = df_sample.dropna()
    return df_sample
