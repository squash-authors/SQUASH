import sys
import os
import numpy as np
sys.path.append('SQUASH_BUILD/src')
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed


from src.qsession import QSession as QSession
from src.partitioner import Partitioner as Partitioner
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
def get_partition_pops(path, fname):
    with np.load(os.path.join(path, '') + fname + '.ptnrvars.npz') as data:
        partition_pops = data['PART_POPS']
    return partition_pops
# ----------------------------------------------------------------------------------------------------------------------------------------
def process_partition(partition_id, partition_path, fname, mode, num_partitions, num_part_vecs, num_dimensions,
                      num_attributes, num_blocks, word_size, big_endian, bit_budget, attribute_bit_budget,
                      non_uniform_bit_alloc, design_boundaries):
        
    # Redirect output/error to named file under /logs
    logpath = Path('SQUASH_BUILD/logs/' + fname)
    out = os.path.join(logpath, str(partition_id) + "_" + str(os.getpid()) + ".out")
    sys.stdout = open(out, "a")
    sys.stderr = sys.stdout
       
    session = QSession(path=partition_path, fname=fname, mode=mode, num_partitions=num_partitions, num_vectors=num_part_vecs, num_dimensions=num_dimensions,   \
        num_attributes=num_attributes, num_blocks=num_blocks, word_size=word_size, big_endian=big_endian, bit_budget=bit_budget, \
        attribute_bit_budget=attribute_bit_budget, non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries)        

    print()
    print('RUN PARAMETERS')
    print('--------------')
    print('** PARTITIONED RUN : PARTITION ID ', partition_id, ' **')
    print()
    for item in session.__dict__.items():
        print(item)
    session.run()   
    del session    
    
   
    return partition_id 
# ----------------------------------------------------------------------------------------------------------------------------------------
def main():
   
    #------------------------------
    # Params common across datasets
    #------------------------------
    # Todo: Add Rustam's argparse in this script
    mode                    = 'B'   # Only (B)uild or (R)ebuild for SQUASH_BUILD
    word_size               = 4
    big_endian              = False
    # bit_budget              = 400
    # bit_budget              = 128
    # bit_budget              = 384
    bit_budget              = 512
    # bit_budget              = 1200
    # bit_budget              = 3840
    # bit_budget              = 7680    
    attribute_bit_budget    = 32
    non_uniform_bit_alloc   = True
    design_boundaries       = True
    partitions_root         = 'partitions'
    
    #------------------------------
    # Dataset-specific params
    #------------------------------
    
    # path = Path('SQUASH_BUILD/datasets/histo64i64_12103/')
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_attributes = 4
    # # num_blocks = 7
    # num_blocks = 1
    
    path = Path('SQUASH_BUILD/datasets/siftsmall/')
    fname = 'siftsmall'
    num_partitions = 4
    partitioner_blocks = 1
    num_vectors = 10000
    num_dimensions = 128
    num_attributes = 4  
    num_blocks = 1
    
    # path = Path('SQUASH_BUILD/datasets/ltest/')    
    # fname = 'ltest'
    # num_partitions = 1
    # partitioner_blocks = 1    
    # num_vectors = 50
    # num_dimensions = 128
    # num_attributes = 4    
    # num_blocks = 1

    # path = Path('SQUASH_BUILD/datasets/sift1m/')
    # fname = 'sift1m'
    # num_partitions = 10 
    # partitioner_blocks = 10    
    # num_vectors = 1000000
    # num_dimensions = 128
    # num_attributes = 4      
    # num_blocks = 1  
     
    # path = Path('SQUASH_BUILD/datasets/gist1/')
    # fname = 'gist1m'
    # num_partitions = 20       
    # partitioner_blocks = 10     
    # num_vectors = 1000000
    # num_attributes = 4 
    # num_dimensions = 960
    # num_blocks = 1  

    # path = Path('SQUASH_BUILD/datasets/word2vec/')
    # fname = 'word2vec'
    # num_partitions = 10 
    # partitioner_blocks = 10    
    # num_vectors = 1000000
    # num_dimensions = 300
    # num_attributes = 4      
    # num_blocks = 1  
    # 

    # path = Path('SQUASH_BUILD/datasets/deep10m/')
    # fname = 'deep10m'
    # num_partitions = 10 
    # partitioner_blocks = 10    
    # num_vectors = 10000000
    # num_dimensions = 96
    # num_attributes = 4      
    # num_blocks = 1      

    # print("Checking num_blocks validity")
    if not check_block_count_validity(num_vectors, num_blocks):
        exit(1)
        
    print("Checking partitioner_blocks validity")
    if not check_block_count_validity(num_vectors, partitioner_blocks):
        exit(1)        

    # Run Partitioner if required
    if (mode in ('B', 'R', 'X')) and (num_partitions > 0):
        partitioner = Partitioner(path=path, fname=fname, mode=mode, word_size=word_size, big_endian=big_endian, \
                                 attribute_bit_budget=attribute_bit_budget, non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries, \
                                 partitioner_blocks=partitioner_blocks, num_vectors=num_vectors, num_dimensions=num_dimensions, num_attributes=num_attributes, num_partitions=num_partitions)

        part_pops = partitioner.process()

    elif num_partitions > 0:
        part_pops = get_partition_pops(path, fname)

    # Instantiate session and commence run. If partitioned, run session once per partition
    if num_partitions == 0:
    
        session = QSession(path=path, fname=fname, mode=mode, num_partitions=num_partitions, num_vectors=num_vectors, num_dimensions=num_dimensions, \
            num_attributes=num_attributes, num_blocks=num_blocks, word_size=word_size, big_endian=big_endian, bit_budget=bit_budget, \
            attribute_bit_budget=attribute_bit_budget, non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries)
        
        print()
        print('RUN PARAMETERS')
        print('--------------')
        print('** NON-PARTITIONED RUN **')
        print()
        for item in session.__dict__.items():
            print(item)
        session.run()
    
    else: # Multiple partitions - use MP

        # Set up folder for MP logs
        logpath = Path('SQUASH_BUILD/logs/' + fname)
        if os.path.exists(logpath):
            shutil.rmtree(logpath)
        os.mkdir(logpath)          

        # with ProcessPoolExecutor(max_workers=num_partitions) as executor:
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = []    
            print("Starting Partition-Level Multiprocessing Loop..",flush=True)        
        
            for part_id in range(num_partitions):
                partition_path  = os.path.join(os.path.join(path,partitions_root),str(part_id))
                num_part_vecs   = part_pops[part_id]
                
                futures.append( executor.submit(process_partition, partition_id=part_id, partition_path=partition_path, fname=fname, mode=mode, num_partitions=num_partitions,
                                                num_part_vecs=num_part_vecs, num_dimensions=num_dimensions, num_attributes=num_attributes, num_blocks=num_blocks,
                                                word_size=word_size, big_endian=big_endian, bit_budget=bit_budget, attribute_bit_budget=attribute_bit_budget,
                                                non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries) )  
                
                # process_partition(partition_id=part_id, partition_path=partition_path, fname=fname, mode=mode, num_partitions=num_partitions,
                #                                 num_part_vecs=num_part_vecs, num_dimensions=num_dimensions, num_attributes=num_attributes, num_blocks=num_blocks,
                #                                 word_size=word_size, big_endian=big_endian, bit_budget=bit_budget, attribute_bit_budget=attribute_bit_budget,
                #                                 non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries) 
                
                
            for future in as_completed(futures):
                id = future.result()
                print('Processing completed for Partition ', id)
                
    
if __name__ == "__main__":
    main()