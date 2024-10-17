import sys
sys.path.append('SQUASH_BUILD/src')
from pathlib import Path

from qsession import QSession as QSession

def calc_valid_num_blocks(num_vectors):
    valids = []
    for i in range(20):
        if num_vectors % i == 0:
            valids.append(i)
    print("Valid block counts for " , str(num_vectors) , " vectors: ", str(valids))

    return valids

def check_block_count_validity(num_vectors, num_blocks):
    if num_vectors % num_blocks == 0:
        # print("Valid number of blocks selected: ", str(num_blocks))
        return True
    else:
        print("Invalid number of blocks selected.", str(num_blocks))
        return False

def main():
    
    #------------------------------
    # Params common across datasets
    #------------------------------
    mode                    = 'P'
    create_qhist            = True
    use_qhist               = True
    query_k                 = 10
    query_fname             = None
    qhist_fname             = None
    word_size               = 4
    big_endian              = False
    q_lambda                = 1
    # bit_budget              = 512
    # bit_budget              = 4000
    non_uniform_bit_alloc   = True
    design_boundaries       = True
    dual_phase              = True
    inmem_vaqdata           = 'inmem_columnar'
    relative_dist           = True
    use_bitarrays           = False
        
    #------------------------------
    # Dataset-specific params
    #------------------------------
    
    # path = Path('SQUASH_BUILD/datasets/histo64i64_12103/')
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_attributes = 4
    # num_blocks = 7
    
    path = Path('SQUASH_BUILD/datasets/siftsmall/')
    fname = 'siftsmall'
    num_vectors = 10000
    num_dimensions = 128
    num_attributes = 4    
    num_blocks = 10
    vecs_to_print = [183,4117]
    
    # path = Path('SQUASH_BUILD/datasets/ltest/')    
    # fname = 'ltest'
    # num_vectors = 50
    # num_dimensions = 128
    # num_attributes = 4    
    # num_blocks = 1

    # path = Path('SQUASH_BUILD/datasets/sift1m/')
    # fname = 'sift1m'
    # num_vectors = 1000000
    # num_dimensions = 128
    # # num_attributes = 4    
    # num_blocks = 10
    # # vecs_to_print = [7736, 474709, 125600, 528651, 140765, 655830, 381822]
    vecs_to_print = [387191]
    
    # path = Path('SQUASH_BUILD/datasets/gist1m/')
    # fname = 'gist1m'
    # num_vectors = 1000000
    # num_dimensions = 960
    # num_attributes = 4    
    # num_blocks = 10
            

    # print("Checking num_blocks validity")
    if not check_block_count_validity(num_vectors, num_blocks):
        exit(1)

    # Instantiate session and commence run
    session = QSession(path=path, fname=fname, mode=mode, query_k=query_k, query_fname=query_fname, \
        num_vectors=num_vectors, num_dimensions=num_dimensions, num_attributes=num_attributes, num_blocks=num_blocks, word_size=word_size, \
        big_endian=big_endian, bit_budget=bit_budget, non_uniform_bit_alloc=non_uniform_bit_alloc, \
        design_boundaries=design_boundaries, inmem_vaqdata=inmem_vaqdata, vecs_to_print=vecs_to_print)

    print()
    print('RUN PARAMETERS')
    print('--------------')
    for item in session.__dict__.items():
        print(item)
    
    session.run()
    
if __name__ == "__main__":
    main()