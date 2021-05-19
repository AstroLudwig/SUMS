from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from astropy.io import fits

import os
import pandas as pd 

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


def get_file(id,segment,filter,extension,gal):

    calib_key = '1923wOC_XwbGt0L9o9heLBwF7mU5KPGLO'
    uv_key = '1y2th7NKfHS5fMk3poBq_vayxwLV9l2mB'
    optical_key = '1LcFrFOqN1pBaewHwl7_jZ89BBjpbkgDV'

    file_list = drive.ListFile({'q': f"'{uv_key}' in parents and trashed=false"}).GetList()

    
    if gal == 'lmc':
        gal_id = file_list[0]['id']
    elif gal == 'smc':
        gal_id = file_list[1]['id']
    else:
        print('error in gal name')

    file_list = drive.ListFile({'q': f"'{gal_id}' in parents and trashed=false"}).GetList()
    fname = f'sw000{id}00{segment}{filter}_sk_{id}_{segment}_{extension}'
    img_name = fname + '.new'
    cat_name = fname + '.full.dat'
    
    for file1 in file_list:
        if file1['title'] == img_name:
            print('Found Image')
            sname = f'temp_{id}_{segment}_{filter}_{extension}.new'
            file1.GetContentFile(sname)
            hdr = fits.open(sname)
            #os.remove(sname)
        if file1['title'] == cat_name:
            print('Found Catalog')
            cname = f'temp_{id}_{segment}_{filter}_{extension}.full.dat'
            file1.GetContentFile(cname)
            #cat=cname
            cat = pd.read_csv(cname,delimiter='\s+',names=['RAhr','DEdeg','Umag','e_Umag','Bmag','e_Bmag','Vmag','e_Vmag','Imag','e_Imag','Flag','Jmag','e_Jmag','Hmag','e_Hmag','Ksmag','e_Ksmag'])
            #os.remove(cname)
    
    return hdr,cat