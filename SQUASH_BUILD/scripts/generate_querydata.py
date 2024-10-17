import sys
sys.path.append('SQUASH_BUILD/src')
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import os
import random
import csv
import shutil
import timeit
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed

class GenerateQueriesMP:
    
    GT_ENTRIES  = 100
    NUM_BINS    = 40
    
    def __init__(self, path, fname, num_vectors=None, num_dimensions=None, num_attributes=None, sample_inds=None, attr_selectivity_perc=None, allocators_root=None, allocator_id=None):
        
        # Params
        self.path                       = path
        self.fname                      = fname
        self.num_vectors                = num_vectors
        self.num_dimensions             = num_dimensions
        self.num_attributes             = num_attributes
        self.sample_inds                = sample_inds
        self.attr_selectivity_perc      = attr_selectivity_perc  
        self.allocators_root            = allocators_root      
        self.allocator_id               = allocator_id
        
        # Other
        self.allocator_path             = None
        self.num_samples                = None
        self.vec_file                   = None
        self.attr_file                  = None
        self.querydata_file             = None
        self.num_words_per_block        = 0
        self.num_vectors_per_block      = 0        
        self.queries                    = None
        self.distances                  = None
        self.predicate_candidates       = None
        self.predicate_sets             = None
        self.gtraw                      = None  
        self.gtraw_dists                = None
        self.gtattr                     = None
        self.gtattr_dists               = None
        self.word_size                  = 4
        self.big_endian                 = False
        self.num_bins                   = 20   
    #----------------------------------------------------------------------------------------------------------------------------------------   
    def initialise(self):
        np.set_printoptions(suppress=True, threshold=np.inf, linewidth=140)

        self.vec_file                   = os.path.join(self.path, '') + self.fname
        self.attr_file                  = os.path.join(self.path, '') + self.fname + ".af"
        self.allocator_path             = os.path.join(self.path, self.allocators_root, str(self.allocator_id))  
        self.querydata_file             = os.path.join(self.allocator_path, '') + self.fname + "_qry" 
        self.num_samples                = self.sample_inds.shape[0] 
        self.queries                    = np.zeros((self.num_samples, self.num_dimensions+1), dtype=np.float32)

        self.gtraw                      = np.full((self.num_samples, GenerateQueriesMP.GT_ENTRIES+1), -1, dtype=np.uint32)    # Extra 1 to match irisa format which has a 4 byte prefix(value 100)
        self.gtraw_dists                = np.full((self.num_samples, GenerateQueriesMP.GT_ENTRIES), -1, dtype=np.float32)
        self.distances                  = np.full((self.num_samples, self.num_vectors), -1, dtype=np.float32)
                
        self.gtattr                     = np.full((self.num_samples, GenerateQueriesMP.GT_ENTRIES+1), -1, dtype=np.uint32)
        self.gtattr_dists               = np.full((self.num_samples, GenerateQueriesMP.GT_ENTRIES), -1, dtype=np.float32)   
        self.predicate_sets             = np.empty((self.num_samples, (self.num_attributes * 4)), dtype='U10')           
    #----------------------------------------------------------------------------------------------------------------------------------------  
    def generate_vectors(self):
        vec_size = self.num_dimensions + 1
        offset = 0
        with open(self.vec_file, mode="rb") as f:
            
            while True:
                f.seek(offset, os.SEEK_SET) 
                vec = np.fromfile(file=f, count=vec_size, dtype=np.float32)
                if vec.size > 0:
                    yield vec
                    offset += (vec_size * self.word_size)
                else:
                    break
    #----------------------------------------------------------------------------------------------------------------------------------------
    def generate_attribute_set(self):
        block_idx = 0      
        with open(self.attr_file, mode="rb") as f:
            while True:
                f.seek(self.num_vectors * block_idx * self.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                if self.big_endian:
                    block = np.fromfile(file=f, count=self.num_vectors, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.num_vectors, dtype=np.float32)
                if block.size > 0:
                    block = np.reshape(block, (self.num_vectors, 1), order="C")  # Order F to mirror MATLAB  NOT SURE IF NEEDED!
                    yield block
                    block_idx +=1
                else:
                    break     
    #---------------------------------------------------------------------------------------------------------------------------------------- 
    def read_vector(self, vector_idx):
        # vector_idx is the index of the required vector, assuming zero counting
        offset = np.int64( np.int64(vector_idx) * np.int64(self.num_dimensions + 1) * np.int64(self.word_size) )
        with open(self.vec_file, mode="rb") as f:
            f.seek(offset, os.SEEK_SET)
            vector = np.fromfile(f, count=(self.num_dimensions + 1), dtype=np.float32)
        return vector
    #----------------------------------------------------------------------------------------------------------------------------------------
    def build_queryset(self):
        read_count = 0
        for ind in self.sample_inds:
            vec = self.read_vector(ind)
            self.queries[read_count,:] = vec
            read_count += 1
            
        print()
        print("Query Vectors Generated : ", str(read_count))
        print()
    #----------------------------------------------------------------------------------------------------------------------------------------     
    def build_ground_truth_raw(self):
        
        print('Allocator : ', self.allocator_id, ' Build Ground Truths (Raw) begins')
        gene_ds = self.generate_vectors()
        vecno = 0
        for vector in gene_ds:
            self.distances[:,vecno] = np.sqrt(np.sum(np.square(np.subtract(self.queries,vector)),axis=1))
            vecno += 1
            
        sortinds = np.argsort(self.distances,axis=1)
        self.gtraw[:,1:GenerateQueriesMP.GT_ENTRIES+1] = sortinds[:,0:GenerateQueriesMP.GT_ENTRIES]
        self.gtraw[:,0] = 100
        for i in range(self.num_samples):
            self.gtraw_dists[i] = self.distances[i][self.gtraw[i,1:GenerateQueriesMP.GT_ENTRIES+1]]

        print('Allocator : ', self.allocator_id, ' Build Ground Truths (Raw) completed')

        # print('Ground Truths (Raw)')
        # for j in range(self.num_samples):
        #     print(self.gtraw[j,1:11])
        #     print(self.gtraw_dists[j,0:10])
        #     print()
    #----------------------------------------------------------------------------------------------------------------------------------------
    def calc_preds_and_candidates(self):
        
        # Calc numbers of entries to select from each attribute set
        selectivity             = self.attr_selectivity_perc / 100
        vector_count            = self.num_vectors
        target_count            = np.uint32(np.ceil(vector_count * selectivity))
        sel_factor              = np.power(selectivity, (1/self.num_attributes))
        attribute_target_counts = np.zeros(self.num_attributes, dtype=np.uint32)
        vector_mask             = np.ones(self.num_vectors, dtype=np.uint8)
        
        for i in range(self.num_attributes):
            to_take                     = vector_count * sel_factor
            attribute_target_counts[i]  = to_take
            # vector_count              -= to_take
            vector_count                *= sel_factor
        
        # Find predicate ranges
        operators       = ['<','>','B']
        attr_no         = 0
        attr_gene       = self.generate_attribute_set()
        predicate_set   = []
        for aset in attr_gene:
            
            rcount      = 0
            operator    = random.choice(operators)
            hist, bins  = np.histogram(aset[vector_mask==1], bins=GenerateQueriesMP.NUM_BINS)
            
            if operator == '<':
                ind = 0
                while (rcount < attribute_target_counts[attr_no]) and (ind < hist.shape[0]):
                    rcount  += hist[ind]
                    ind     += 1
                end = bins[ind]
                predicate_set.extend([str(attr_no), operator, str(end), '0'])
                attr_no += 1
                matching_vecs = np.where((aset < end),1,0).ravel()
                vector_mask = np.bitwise_and(vector_mask, matching_vecs)

            elif operator == '>':
                ind = GenerateQueriesMP.NUM_BINS - 1
                while rcount < attribute_target_counts[attr_no]:
                    rcount  += hist[ind]
                    ind     -= 1
                start = bins[ind]
                predicate_set.extend([str(attr_no), operator, str(start), '0'])
                attr_no += 1
                matching_vecs = np.where((aset > start),1,0).ravel()
                vector_mask = np.bitwise_and(vector_mask, matching_vecs)                
                
            elif operator == 'B':
                start_ind = np.uint32(np.floor(GenerateQueriesMP.NUM_BINS / 2))
                while True:
                    ind = start_ind
                    start   = bins[ind]
                    while rcount < attribute_target_counts[attr_no]:
                        rcount  += hist[ind]
                        ind     += 1           
                        if ind > GenerateQueriesMP.NUM_BINS -1:
                            break
                    if rcount >= attribute_target_counts[attr_no]:
                        end = bins[ind]
                        predicate_set.extend([str(attr_no), operator, str(start), str(end)])
                        attr_no += 1
                        condition1 = aset > start
                        condition2 = aset <= end
                        matching_vecs = np.where(condition1 & condition2,1,0).ravel()
                        vector_mask = np.bitwise_and(vector_mask, matching_vecs)                         
                        break
                    start_ind -= 1  
                    rcount = 0    
                    if start_ind < 0:
                        print('build_predicate_file : Cannot retrieve enough vectors for operator B - Aborting!')      
                        exit(1)
                    
        candidates = np.where(vector_mask==1)[0]   
        return predicate_set, candidates
    #----------------------------------------------------------------------------------------------------------------------------------------
    def build_ground_truth_attr(self):
        
        self.bgta_start_dt = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Allocator : ', self.allocator_id, ' Build Ground Truth (Attributed) begins at : ', str(current_time))
        
        prefix = np.array([100], dtype=np.uint32)
        chkpt_size = self.num_samples // 10
        for qno in range(self.num_samples):
            
            predicate_set, candidates   = self.calc_preds_and_candidates()
            self.distances              = np.full((candidates.shape[0]), -1, dtype=np.float32) 
            vecno = 0
            for vecid in candidates:
                vector = self.read_vector(vecid)
                self.distances[vecno] = np.sqrt(np.sum(np.square(np.subtract(self.queries[qno],vector))))
                vecno += 1
                
            sortinds                = np.argsort(self.distances)
            gtattr                  = np.uint32(candidates[sortinds][0:GenerateQueriesMP.GT_ENTRIES])
            gtattr                  = np.concatenate((prefix, gtattr),dtype=np.uint32)
            gtattr_dists            = self.distances[sortinds][0:GenerateQueriesMP.GT_ENTRIES]
            self.gtattr[qno]        = gtattr
            self.gtattr_dists[qno]  = gtattr_dists                
            self.predicate_sets[qno] = np.array(predicate_set, dtype='U10')

            # print(predicate_set)
            # print(gtattr[1:11])
            # print(gtattr_dists[0:10])
            # print()        
            
            if qno % chkpt_size == 0:
                dt = datetime.now()
                current_time = dt.strftime("%H:%M:%S")
                print('Allocator : ', self.allocator_id, ' Build Ground Truth (Attributed) : Processing Query ', qno, ' at ', current_time)
            
            
        self.bgta_end_dt = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print('Allocator : ', self.allocator_id, ' Build Ground Truth (Attributed) ends at : ', str(current_time), ' Elapsed : ', str(self.bgta_end_dt - self.bgta_start_dt) )
            
    #----------------------------------------------------------------------------------------------------------------------------------------                    
    def write_querydata_file(self):
        np.savez(   self.querydata_file,
                    INDS            = self.sample_inds,
                    QUERIES         = self.queries,
                    PRED_SETS       = self.predicate_sets,
                    GT_RAW          = self.gtraw,
                    GT_RAW_DISTS    = self.gtraw_dists,
                    GT_ATTR         = self.gtattr,
                    GT_ATTR_DISTS   = self.gtattr_dists )
    #----------------------------------------------------------------------------------------------------------------------------------------                    
    def read_querydata_file(self):        
        with np.load(self.querydata_file + '.npz') as querydata:
                self.sample_inds                = querydata['INDS']
                self.queries                    = querydata['QUERIES']
                self.predicate_sets             = querydata['PRED_SETS']
                self.gtraw                      = querydata['GT_RAW']
                self.gtraw_dists                = querydata['GT_RAW_DISTS']  
                self.gtattr                     = querydata['GT_ATTR']  
                self.gtattr_dists               = querydata['GT_ATTR_DISTS']            
    #----------------------------------------------------------------------------------------------------------------------------------------    
    def process(self):
        print()
        print("Generate Querydata MP Session Begins")
        print("====================================")
        now = datetime.now()
        self.gqmp_start_dt = timeit.default_timer()        
        current_time = now.strftime("%H:%M:%S")
        print(' Allocator : ', self.allocator_id, ' Generate Querydata MP Session Start Time : ', str(current_time))

        self.initialise()
        self.build_queryset()
        self.build_ground_truth_raw()
        self.build_ground_truth_attr()
        self.write_querydata_file()
        
        self.gqmp_end_dt = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(' Allocator : ', self.allocator_id, ' Generate Querydata MP Session End Time : ', current_time, 'Elapsed : ', str(self.gqmp_end_dt - self.gqmp_start_dt))
        print()
    #----------------------------------------------------------------------------------------------------------------------------------------
    # END CLASS DEFINITION
    #----------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------                        
# NON-CLASS FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------------------------                    
def generate_queries (path, fname, num_vectors, num_dimensions, num_attributes, sample_inds, attr_selectivity_perc, allocators_root, allocator_id):
    
    gqmp = GenerateQueriesMP(path, fname, num_vectors, num_dimensions, num_attributes, sample_inds, attr_selectivity_perc, allocators_root, allocator_id)
    gqmp.process()

#----------------------------------------------------------------------------------------------------------------------------------------                    
def recreate_allocator_dirs(path, allocators_root, num_allocators):
    # Clear old allocators folders and data (if present)
    allocators_path = os.path.join(path, allocators_root)
    if os.path.exists(allocators_path):
        shutil.rmtree(allocators_path)

    # Create new allocators folders
    os.mkdir(allocators_path)
    for allocator_no in range(num_allocators):
        allocator_dir = os.path.join(allocators_path, str(allocator_no))
        os.mkdir(allocator_dir)
#---------------------------------------------------------------------------------------------------------------------------------------- 
def generate_sample_inds(num_vectors, num_samples):
    all_vector_inds = list(np.arange(0,num_vectors, dtype=np.uint32))
    sel = random.sample(all_vector_inds, num_samples)
    sel.sort()
    sample_inds = np.array(sel, dtype=np.uint32)

    print()
    print("Random sample of vectors : Selecting:")
    print(sample_inds)
    print()
   
    return sample_inds
#----------------------------------------------------------------------------------------------------------------------------------------
def build_global_querydata_file(path, fname, num_allocators, allocators_root):
    
    print('Building Global Querydata file from Allocators subsets')
    
    for a_id in range(num_allocators):
        querydata_fname  = os.path.join(path, allocators_root, str(a_id), fname) + '_qry.npz'
        # query_fname     = os.path.join(allocator_path, '') + fname + '_qry.npz'
        with np.load(querydata_fname) as querydata:
            sample_inds     = querydata['INDS']
            queries         = querydata['QUERIES']
            predicate_sets  = querydata['PRED_SETS']
            gtraw           = querydata['GT_RAW']
            gtraw_dists     = querydata['GT_RAW_DISTS']  
            gtattr          = querydata['GT_ATTR']  
            gtattr_dists    = querydata['GT_ATTR_DISTS'] 
        
        if a_id == 0:
            all_sample_inds     = sample_inds
            all_queries         = queries
            all_predicate_sets  = predicate_sets
            all_gtraw           = gtraw
            all_gtraw_dists     = gtraw_dists
            all_gtattr          = gtattr
            all_gtattr_dists    = gtattr_dists
        else:    
            all_sample_inds     = np.hstack((all_sample_inds, sample_inds))    
            all_queries         = np.vstack((all_queries, queries))
            all_predicate_sets  = np.vstack((all_predicate_sets, predicate_sets))
            all_gtraw           = np.vstack((all_gtraw, gtraw))
            all_gtraw_dists     = np.vstack((all_gtraw_dists, gtraw_dists))
            all_gtattr          = np.vstack((all_gtattr, gtattr))
            all_gtattr_dists    = np.vstack((all_gtattr_dists, gtattr_dists))            

    global_query_fname = os.path.join(path,'') + fname + '_qry'
    np.savez(   global_query_fname,
                INDS            = all_sample_inds,
                QUERIES         = all_queries,
                PRED_SETS       = all_predicate_sets,
                GT_RAW          = all_gtraw,
                GT_RAW_DISTS    = all_gtraw_dists,
                GT_ATTR         = all_gtattr,
                GT_ATTR_DISTS   = all_gtattr_dists )    

#----------------------------------------------------------------------------------------------------------------------------------------    
def duplicate_allocators(path, fname, num_allocators, allocators_root, dup_factor):

    for df in range(dup_factor - 1):
        for a_id in range(num_allocators):
            target_a_id = (df * num_allocators) + (a_id + num_allocators)
            source  = os.path.join(path, allocators_root, str(a_id))
            dest    = os.path.join(path, allocators_root, str(target_a_id))
            print('Duplicating folder : ', source, ' To folder : ', dest)
            new_path = shutil.copytree(source, dest)
#----------------------------------------------------------------------------------------------------------------------------------------    
def create_query_subsets(path, fname, num_allocators, allocators_root, dup_factor, caching):
    
    global_querydata_fname = os.path.join(path, '') + fname + '_qry.npz'
    with np.load(global_querydata_fname) as querydata:
        
        if dup_factor > 1 and caching == False:
            print('Duplicating queries using duplication factor ', dup_factor, ' - NOT FOR USE WITH CACHING!!')        
            sample_inds                = np.tile(querydata['INDS']          , dup_factor-1)
            queries                    = np.tile(querydata['QUERIES']       , (dup_factor-1,1))
            predicate_sets             = np.tile(querydata['PRED_SETS']     , (dup_factor-1,1))
            gtraw                      = np.tile(querydata['GT_RAW']        , (dup_factor-1,1))
            gtraw_dists                = np.tile(querydata['GT_RAW_DISTS']  , (dup_factor-1,1))
            gtattr                     = np.tile(querydata['GT_ATTR']       , (dup_factor-1,1))
            gtattr_dists               = np.tile(querydata['GT_ATTR_DISTS'] , (dup_factor-1,1))             
       
        else:
            sample_inds                = querydata['INDS']
            queries                    = querydata['QUERIES']
            predicate_sets             = querydata['PRED_SETS']
            gtraw                      = querydata['GT_RAW']
            gtraw_dists                = querydata['GT_RAW_DISTS']  
            gtattr                     = querydata['GT_ATTR']  
            gtattr_dists               = querydata['GT_ATTR_DISTS']         
            
        print()        

    # Split querysets        
    ind_splits          = np.array_split(sample_inds,    num_allocators)
    query_splits        = np.array_split(queries,        num_allocators)
    pred_splits         = np.array_split(predicate_sets, num_allocators)
    gtraw_splits        = np.array_split(gtraw,          num_allocators)
    gtraw_dists_splits  = np.array_split(gtraw_dists,    num_allocators)
    gtattr_splits       = np.array_split(gtattr,         num_allocators)
    gtattr_dists_splits = np.array_split(gtattr_dists,   num_allocators)
            
    # Where appropriate, duplicate splits for use with caching
    if dup_factor > 1 and caching == True:
        print('Duplicating queries using duplication factor ', dup_factor, ' - FOR USE WITH CACHING!!') 
        for allocator_no in range(num_allocators):
            ind_splits          [allocator_no]  = np.tile(ind_splits            [allocator_no], dup_factor-1),
            query_splits        [allocator_no]  = np.tile(query_splits          [allocator_no], (dup_factor-1,1))
            pred_splits         [allocator_no]  = np.tile(pred_splits           [allocator_no], (dup_factor-1,1))
            gtraw_splits        [allocator_no]  = np.tile(gtraw_splits          [allocator_no], (dup_factor-1,1))
            gtraw_dists_splits  [allocator_no]  = np.tile(gtraw_dists_splits    [allocator_no], (dup_factor-1,1))
            gtattr_splits       [allocator_no]  = np.tile(gtattr_splits         [allocator_no], (dup_factor-1,1))
            gtattr_dists_splits  [allocator_no]  = np.tile(gtattr_dists_splits  [allocator_no], (dup_factor-1,1))
    
    # Output details of split sizes
    split_sizes = np.zeros(num_allocators, dtype=np.uint32)
    total_queries = 0
    for i in range(num_allocators):
        split_sizes[i] = query_splits[i].shape[0]     
        total_queries += split_sizes[i] * dup_factor
        print('Allocator : ', i, ' Num Queries : ', split_sizes[i]*dup_factor)   
    
    print()
    print('Total Queries : ', total_queries)
    
    # Save allocator-level querydata files
    for allocator_no in range(num_allocators):
        alloc_path              = os.path.join(path, allocators_root, str(allocator_no))
        alloc_querydata_file    = os.path.join(alloc_path, '') + fname + "_qry"
        np.savez( alloc_querydata_file, 
                    INDS            = ind_splits            [allocator_no],
                    QUERIES         = query_splits          [allocator_no],
                    PRED_SETS       = pred_splits           [allocator_no],
                    GT_RAW          = gtraw_splits          [allocator_no],
                    GT_RAW_DISTS    = gtraw_dists_splits    [allocator_no],
                    GT_ATTR         = gtattr_splits         [allocator_no],
                    GT_ATTR_DISTS   = gtattr_dists_splits   [allocator_no] )
#----------------------------------------------------------------------------------------------------------------------------------------  
def main():

    # path = Path('SQUASH_BUILD/datasets/histo64i64_12103/')    
    # fname = 'histo64i64_12103_swapped'
    # num_vectors = 12103
    # num_dimensions = 64
    # num_attributes = 4    
    # num_blocks = 7 
    # num_samples = 12
    # attr_selectivity_perc = 8

    # path = Path('SQUASH_BUILD/datasets/siftsmall/')
    # fname = 'siftsmall'    
    # num_vectors = 10000
    # num_dimensions = 128
    # num_attributes = 4
    # num_samples = 1000
    # attr_selectivity_perc = 8    

    path = Path('SQUASH_BUILD/datasets/sift1m/')
    fname = 'sift1m'    
    num_vectors = 1000000
    num_dimensions = 128
    num_attributes = 4
    num_samples = 1000
    attr_selectivity_perc = 8    
    
    # path = Path('SQUASH_BUILD/datasets/gist1m/')
    # fname = 'gist1m'    
    # num_vectors = 1000000
    # num_dimensions = 960
    # num_attributes = 4    
    # num_samples = 1000
    # # attr_selectivity_perc = 8    

    # path = Path('SQUASH_BUILD/datasets/deep10m/')
    # fname = 'deep10m'    
    # num_vectors = 10000000
    # num_dimensions = 96
    # num_attributes = 4
    # num_samples = 1000
    # attr_selectivity_perc = 8      

    
    # Common Paramaters
    mode            = 'R'
    num_allocators  = 10    
    dup_factor      = 1
    caching         = True
    allocators_root = "allocators"
    #
    
    if mode == 'G':         # Generate a new set of querydata
        
        # Get random vector ids and split into suitable chunks
        sample_inds         = generate_sample_inds(num_vectors, num_samples)
        sample_ind_splits   = np.array_split(sample_inds, num_allocators)
        recreate_allocator_dirs(path, allocators_root, num_allocators)

        # Run multiple generators to build the allocator-level query files
        with ProcessPoolExecutor(max_workers=num_allocators) as executor:
            futures = []
            for a_id in range(num_allocators):

                futures.append( executor.submit(generate_queries, path=path, fname=fname, num_vectors=num_vectors, num_dimensions=num_dimensions, \
                                                                num_attributes=num_attributes, sample_inds=sample_ind_splits[a_id], \
                                                                attr_selectivity_perc=attr_selectivity_perc, allocators_root=allocators_root, allocator_id=a_id ) )
                
                # generate_queries(path=path, fname=fname, num_vectors=num_vectors, num_dimensions=num_dimensions, \
                #                  num_attributes=num_attributes, sample_inds=sample_ind_splits[a_id], \
                #                  attr_selectivity_perc=attr_selectivity_perc, allocators_root=allocators_root, allocator_id=a_id)
                                    
            for future in as_completed(futures):
                print('Query Generator Process completed')

        # Build global querydata file from allocator-level datasets
        build_global_querydata_file(path, fname, num_allocators, allocators_root)    
        
        # Duplicate if required
        if dup_factor > 1:
            recreate_allocator_dirs(path, allocators_root, num_allocators)
            create_query_subsets(path, fname, num_allocators, allocators_root, dup_factor, caching)
    
    elif mode == 'R' and num_allocators > 1:       # Reallocate queries to new number of Allocators
        recreate_allocator_dirs(path, allocators_root, num_allocators)
        create_query_subsets(path, fname, num_allocators, allocators_root, dup_factor, caching)

    else:
        print('Invalid Mode -> ', mode, ' or num_allocators -> ', num_allocators)
        exit(1)
        


if __name__ == "__main__":
    main()