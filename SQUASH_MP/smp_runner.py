import sys
import os
import numpy as np
sys.path.append('SQUASH_MP/src')
from pathlib import Path
import shutil
import json

# from src.datamanager import DataManager as DataManager
from src.queryallocator import QueryAllocator as QueryAllocator
from src.coordinator import Coordinator as Coordinator
# ----------------------------------------------------------------------------------------------------------------------------------------
def calc_valid_num_blocks(num_vectors):
    valids = []
    for i in range(20):
        if num_vectors % i == 0:
            valids.append(i)
    print("Valid block counts for " , str(num_vectors) , " vectors: ", str(valids))

    return valids
# ----------------------------------------------------------------------------------------------------------------------------------------
def check_block_count_validity(num_vectors, num_blocks):
    if num_vectors % num_blocks == 0:
        # print("Valid number of blocks selected: ", str(num_blocks))
        return True
    else:
        print("Invalid number of blocks selected.", str(num_blocks))
        return False
    
# ----------------------------------------------------------------------------------------------------------------------------------------
def build_payload(   exp_tag, exp_run, path, fname, mode, query_k, query_batchsize, query_fname,  num_vectors, num_dimensions, \
                     num_attributes, num_blocks, word_size, big_endian, bit_budget, attribute_bit_budget, \
                     precision, inmem_vaqdata, adc_or_sdc, bq_cutoff_perc, check_recall, vecs_to_print, \
                     num_partitions, partitions_root, num_allocators, allocators_root, allocator_id, \
                     fine_tune, centroid_factor, k_factor, bfr, l_max, caching, use_s3, s3_bucket):
        
        if vecs_to_print is None:
            vecs_to_print = []
        
        dmg_params = {  "exp_tag"               : exp_tag,
                        "exp_run"               : str(exp_run),
                        "path"                  : str(path), 
                        "fname"                 : fname,
                        "mode"                  : mode,
                        "query_k"               : str(query_k),
                        "query_batchsize"       : str(query_batchsize),
                        "query_fname"           : query_fname,
                        "num_vectors"           : str(num_vectors),
                        "num_dimensions"        : str(num_dimensions),
                        "num_attributes"        : str(num_attributes),
                        "num_blocks"            : str(num_blocks),
                        "word_size"             : str(word_size),
                        "big_endian"            : big_endian,
                        "bit_budget"            : str(bit_budget),
                        "attribute_bit_budget"  : str(attribute_bit_budget),
                        "precision"             : precision,
                        "inmem_vaqdata"         : inmem_vaqdata,
                        "adc_or_sdc"            : adc_or_sdc,
                        "bq_cutoff_perc"        : str(bq_cutoff_perc),
                        "check_recall"          : check_recall,
                        "num_allocators"        : str(num_allocators),
                        "allocators_root"       : allocators_root,
                        "allocator_id"          : str(allocator_id),
                        "fine_tune"             : fine_tune,
                        "centroid_factor"       : str(centroid_factor),
                        "k_factor"              : str(k_factor),
                        "bfr"                   : str(bfr),
                        "l_max"                 : str(l_max),
                        "level"                 : str(0),       # Not passed in but needed later
                        "vecs_to_print"         : vecs_to_print,
                        "caching"               : caching,
                        "use_s3"                : use_s3,
                        "s3_bucket"             : s3_bucket
                    }
        
        qa_params = {   "num_partitions"  : str(num_partitions), 
                        "partitions_root" : partitions_root
                    }
        
        payload = {    "dmg_params"  : dmg_params,
                       "qa_params"   : qa_params
                }          
        
        return json.dumps(payload)    
# ----------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    # In case not set below
    vecs_to_print = None
   
    #------------------------------
    # Params common across datasets
    #------------------------------
    # Todo: Add Rustam's argparse in this script
    exp_tag                 = 'tag'
    exp_run                 = 0   
    mode                    = 'A'   # Only Query(A)llocator or (P)Rint_Vector_Details for squash_run (at this level)
    query_k                 = 10
    query_batchsize         = 50
    query_fname             = None
    word_size               = 4
    big_endian              = False
    # bit_budget              = 128
    # bit_budget              = 256    
    # bit_budget              = 384
    bit_budget              = 512    
    # bit_budget              = 1024
    # bit_budget              = 1200    
    # bit_budget              = 3840
    # bit_budget              = 7680    
    attribute_bit_budget    = 32
    # precision               = 'approx_lb'    
    precision               = 'approx_bqlb'
    # precision               = 'exact'
    inmem_vaqdata           = 'inmem_columnar'
    adc_or_sdc              = 'adc'
    bq_cutoff_perc          = 10
    check_recall            = True
    partitions_root         = 'partitions'
    num_allocators          = 10
    allocators_root         = 'allocators'
    allocator_id            = 0
    fine_tune               = True
    centroid_factor         = 1.3
    k_factor                = 2
    bfr                     = 10
    l_max                   = 1
    caching                 = False
    use_s3                  = False
    s3_bucket               = 'bucket_name'
        
    
    #------------------------------
    # Dataset-specific params
    #------------------------------
    
    # path = Path('SQUASH_MP/datasets/siftsmall/')
    # fname = 'siftsmall'
    # num_partitions = 4
    # num_vectors = 10000
    # num_dimensions = 128
    # num_attributes = 4
    # # num_blocks = 10
    # num_blocks = 1
    # vecs_to_print = [5911,31,7177,7177]

    path = Path('SQUASH_MP/datasets/sift1m/')
    fname = 'sift1m'
    num_partitions = 10   
    num_vectors = 1000000
    num_dimensions = 128
    num_attributes = 4
    num_blocks = 1
    vecs_to_print = [266837, 266670, 267272]

    # path = Path('SQUASH_MP/datasets/gist1m/')
    # fname = 'gist1m'
    # num_partitions = 10    
    # num_vectors = 1000000
    # num_attributes = 4
    # num_dimensions = 960
    # num_blocks = 1

    # path = Path('SQUASH_MP/datasets/deep10m/')
    # fname = 'deep10m'
    # num_partitions = 10   
    # num_vectors = 10000000
    # num_dimensions = 96
    # num_attributes = 4
    # num_blocks = 1
    # vecs_to_print = [266837, 266670, 267272]


    # print("Checking num_blocks validity")
    if not check_block_count_validity(num_vectors, num_blocks):
        exit(1)

    # Build Payload
    payload = build_payload(exp_tag=exp_tag, exp_run=exp_run, path=path, fname=fname, mode=mode, query_k=query_k, query_batchsize=query_batchsize, query_fname=query_fname,  \
        num_vectors=num_vectors, num_dimensions=num_dimensions, num_attributes=num_attributes, num_blocks=num_blocks, word_size=word_size, \
        big_endian=big_endian, bit_budget=bit_budget, attribute_bit_budget=attribute_bit_budget, precision=precision, inmem_vaqdata=inmem_vaqdata, \
        adc_or_sdc=adc_or_sdc, bq_cutoff_perc=bq_cutoff_perc, check_recall=check_recall, vecs_to_print=vecs_to_print, num_partitions=num_partitions, \
        partitions_root=partitions_root, num_allocators=num_allocators, allocators_root=allocators_root, allocator_id=allocator_id, fine_tune=fine_tune, \
        centroid_factor=centroid_factor, k_factor=k_factor, bfr=bfr, l_max=l_max, caching=caching, use_s3=use_s3, s3_bucket=s3_bucket)

   
    # Instantiate and run QueryAllocator or Coordinator
    if mode == 'P': 
        queryallocator = QueryAllocator(payload=payload)
        queryallocator.prvd_processor()
    elif num_allocators == 1:
        queryallocator = QueryAllocator(payload=payload)
        queryallocator.allocate()
    else:
        coordinator = Coordinator(payload=payload)        
        coordinator.coordinate()
    
if __name__ == "__main__":
    main()
