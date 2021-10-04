from astropy.io import fits
import numpy as np 
import wget
import os
import time

#####################################################################

# Download Files From: 
# https://www.swift.ac.uk/swift_live/index.php#advanced


#####################################################################

# UVOT Filters
filters = ["um2","uw1","uw2"]

# Observations Ids for SMC_# Survey
smc_observation_id = np.arange(40415,40465,1).astype(str)

# Observations Ids for LMC_Survey_# Survey
lmc_observation_id = np.arange(45422,45587,1).astype(str)

# Function to download an observation
def wget_file(observation_id,uvfilter,path,imtype='sk'):
    
    ext = ".img.gz"
    sep = "_"


    found_urls = []

    for segment in np.arange(0,20).astype(str) : 
        url = "https://www.swift.ac.uk/archive/reproc/000%s00%s/uvot/image" % (observation_id,segment)
        filename = "/sw000%s00%s%s_%s" % (observation_id,segment,uvfilter,imtype)
        save = path+filename+sep+observation_id+sep+segment+ext
        download = url+filename+ext
        print(download)    
        if os.path.exists(save):
            print("File Already Exists")
            return
        
        try :                    
            wget.download(download,out=save)
            found_urls.append(url)
            
            print(f"Segment Saved: {segment}")
        except :
            #print(f"Failed: {download}") # For debugging
            pass
    
    if len(found_urls) > 1 :
        print(f"Found multiple files for id: {observation_id}")
        

# Download all data        
run_all = False
start = time.time()
if run_all:
    for galaxy,galaxy_id in zip(['/SMC','/LMC'],[smc_observation_id,lmc_observation_id]):
        if os.path.exists(galaxy):
            print("Galactic Folder Exists")
        else:
            os.mkdir(galaxy)
            print(f"{galaxy} folder created.")
        for object_id in galaxy_id:
            print(f"Downloading {object_id}")
            
            for uv_filter in filters:
                wget_file(object_id,uv_filter,galaxy)
    
    print(f"Task completed in {str((time.time - start)/60)} minutes")

# Download one obsid
run_one = True
if run_one:
    obsid = str(32214)
    wget_file(obsid,'um2','SMC/')
    wget_file(obsid,'uw2','SMC/')
    wget_file(obsid,'uw1','SMC/')