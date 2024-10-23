import sys
import os
from pathlib import Path
import shutil
import json
import time
import boto3
import botocore
from botocore.exceptions import ClientError

s3_client = boto3.client('s3', region_name='eu-west-1') 
# ----------------------------------------------------------------------------------------------------------------------------------------
def s3_upload(bucket, s3_fname, local_fname):
    global s3_client    
    try:
        response = s3_client.upload_file(local_fname, bucket, s3_fname)
    except ClientError as err:
        print('Error in s3_upload : Filename ', local_fname, ' Error Code : ', err)
        return False
    return True
# ---------------------------------------------------------------------------------------------------------------------------------------- 
def s3_download(bucket, s3_fname, local_fname):
    global s3_client
    dirname = os.path.dirname(local_fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    try:
        response = s3_client.download_file(bucket, s3_fname, local_fname)
    except ClientError as err:
        print('Error in s3_download : s3 file : ', s3_fname, ' Local file : ', local_fname, ' Error Code : ', err)
        return False
    except FileNotFoundError as err:
        print('Error in s3_download : s3 file : ', s3_fname, ' Local file : ', local_fname, ' Error Code : ', err)
        return False
    return True
# ---------------------------------------------------------------------------------------------------------------------------------------- 
def run_uploads(path, fname, bucket, partitions_root, allocators_root):
    
    # # Upload qavars file
    # key = os.path.join(path,'') + fname + '.qavars.npz'
    # if s3_upload(bucket, key, key):
    #     print('Uploaded file : ', key)
    # else:
    #     exit(1)
        
    # Upload queryset files for all allocators
    print()
    allocator_id = 0
    while True:
        allocator_path = os.path.join(path, allocators_root, str(allocator_id), '')
        if not os.path.exists(allocator_path):
            break
        key = allocator_path + fname + '_qry.npz'
        if s3_upload(bucket, key, key):
            print('Uploaded file : ', key)
        else:
            exit(1)
        allocator_id += 1        
            
    # # Upload qpvars files for all partitions
    # print()
    # partition_id = 0
    # while True:
    #     partition_path = os.path.join(path, partitions_root, str(partition_id), '')
    #     if not os.path.exists(partition_path):
    #         break
    #     key = partition_path + fname + '.qpvars.npz'
    #     if s3_upload(bucket, key, key):
    #         print('Uploaded file : ', key)
    #     else:
    #         exit(1)
    #     partition_id += 1      
        
    print()
    print('s3 dataset uploads completed OK!')      
    
# ---------------------------------------------------------------------------------------------------------------------------------------- 
def run_downloads(path, fname, bucket, partitions_root, allocators_root, local_dir):
    
    # Download qavars file
    s3_fname    = os.path.join(path,'') + fname + '.qavars.npz'
    local_fname = os.path.join(local_dir, path, '') + fname + '.qavars.npz'
    if s3_download(bucket, s3_fname, local_fname):
        print('Downloaded file : ', s3_fname , " To ", local_fname)
    else:
        print('run_downloads -> Failed to download s3 file : ', s3_fname, ' To : ', local_fname)
        exit(1)
        
    # Download queryset files for all allocators
    print()
    allocator_id = 0
    while True:
        s3_allocator_path       = os.path.join(path, allocators_root, str(allocator_id), '')
        local_allocator_path    = os.path.join(local_dir, s3_allocator_path)
        s3_fname                = s3_allocator_path + fname + '_qry.npz'
        local_fname             = local_allocator_path + fname + '_qry.npz'    
        if s3_download(bucket, s3_fname, local_fname):
            print('Downloaded file : ', s3_fname, ' To : ', local_fname)
        else:
            break
        allocator_id += 1        
            
    # Download qpvars files for all partitions
    print()
    partition_id = 0
    while True:
        s3_partition_path       = os.path.join(path, partitions_root, str(partition_id), '')
        local_partition_path    = os.path.join(local_dir, s3_partition_path)
        s3_fname                = s3_partition_path + fname + '.qpvars.npz'
        local_fname             = local_partition_path + fname + '.qpvars.npz'        
        if s3_download(bucket, s3_fname, local_fname):
            print('Downloaded file : ', s3_fname, ' To : ', local_fname)
        else:
            break
        partition_id += 1      
        
    print()
    print('s3 dataset downloads completed OK!')     
# ---------------------------------------------------------------------------------------------------------------------------------------- 
def main():

    # path            = Path('datasets/sift1m_bb512_p10_csuint8_30iter/')
    # path            = Path('datasets/gist1m_bb3840_p10_csuint8_30iter/')
    path            = Path('datasets/sift10m_bb512_p20_csuint8_30iter/')
    # path            = Path('datasets/deep10m_bb384_p20_csuint8_30iter/')

    local_dir       = Path('s3_download_tests/')    
    # fname           = 'sift1m'
    # fname           = 'gist1m'
    fname           = 'sift10m'
    # fname           = 'deep10m'

    bucket          = 'squash-data-bucket'
    partitions_root = 'partitions'
    allocators_root = 'allocators'

    run_uploads(path, fname, bucket, partitions_root, allocators_root)      
    # run_downloads(path, fname, bucket, partitions_root, allocators_root, local_dir)          

# ---------------------------------------------------------------------------------------------------------------------------------------- 
if __name__ == "__main__":
    main()