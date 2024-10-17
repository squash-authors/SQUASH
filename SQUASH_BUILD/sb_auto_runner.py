import sys
import os
import numpy as np
sys.path.append('SQUASH_BUILD/src')
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed
import argparse
import json


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
def main(args):
   
    run_params_file = "SQUASH_BUILD/datasets/" + args.run_dir + "/params.json"

    with open(run_params_file, mode="r") as f:
        params = json.load(f)
    
    mode = params["mode"]
    word_size = int(params["word_size"])
    big_endian = params["big_endian"]
    bit_budget = int(params["bit_budget"])
    attribute_bit_budget = int(params["attribute_bit_budget"])
    non_uniform_bit_alloc = params["non_uniform_bit_alloc"]
    design_boundaries = params["design_boundaries"]
    partitions_root = params["partitions_root"]
    path = Path(params["path"])
    fname = params["fname"]
    num_partitions = int(params["num_partitions"])
    partitioner_blocks = int(params["partitioner_blocks"])
    num_vectors = int(params["num_vectors"])
    num_dimensions = int(params["num_dimensions"])
    num_attributes = int(params["num_attributes"])
    num_blocks = int(params["num_blocks"])
    stages = str(params["stages"]) # P: Partitioner, I: Indexer, A: All(er)


    # print("Checking num_blocks validity")
    if not check_block_count_validity(num_vectors, num_blocks):
        exit(1)
        
    print("Checking partitioner_blocks validity")
    if not check_block_count_validity(num_vectors, partitioner_blocks):
        exit(1)        

    if stages in ('P', 'A'):

        # Run Partitioner if required
        if (mode in ('B', 'R')) and (num_partitions > 0):
            partitioner = Partitioner(path=path, fname=fname, mode=mode, word_size=word_size, big_endian=big_endian, \
                                    attribute_bit_budget=attribute_bit_budget, non_uniform_bit_alloc=non_uniform_bit_alloc, design_boundaries=design_boundaries, \
                                    partitioner_blocks=partitioner_blocks, num_vectors=num_vectors, num_dimensions=num_dimensions, num_attributes=num_attributes, num_partitions=num_partitions)

            part_pops = partitioner.process()

        # part_pops = get_partition_pops(path, fname)

    # elif num_partitions > 0:
    #     part_pops = get_partition_pops(path, fname)

    if stages in ('I', 'A'):

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
    parser = argparse.ArgumentParser(description='SQUASH')
    parser.add_argument('--run_dir', action="store", dest='run_dir', default="")
    args = parser.parse_args()
    main(args)
    