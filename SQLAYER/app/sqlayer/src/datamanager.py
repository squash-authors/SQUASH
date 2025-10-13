import numpy as np
import os
import json
import math
import timeit
import csv

from datetime import datetime
# from bitarray import bitarray, util
from pathlib import Path
import faiss

class DataManager:

    DEBUG = False
    BITWISE_CONTAINER_DATATYPE = np.uint8
    COMPRESSION_LEVEL           = 3    

    def __init__(self, params):
        
        self.params                     = params
        
        # Parameters (to be unpacked from params)
        self.exp_tag                    = None
        self.exp_run                    = None   
        self.path                       = None
        self.fname                      = None
        self.mode                       = None
        self.query_k                    = None
        self.query_batchsize            = None
        self.query_fname                = None
        self.num_vectors                = None
        self.num_dimensions             = None
        self.num_attributes             = None
        self.num_blocks                 = None
        self.word_size                  = None
        self.datatype                   = None        
        self.big_endian                 = None
        self.bit_budget                 = None
        self.attribute_bit_budget       = None
        self.precision                  = None
        self.inmem_vaqdata              = None
        self.adc_or_sdc                 = None
        self.bq_cutoff_perc             = None
        self.check_recall               = None
        self.num_allocators             = None
        self.allocators_root            = None
        self.allocator_id               = None
        self.fine_tune                  = None
        self.centroid_factor            = None
        self.k_factor                   = None
        self.bfr                        = None
        self.l_max                      = None
        self.level                      = None
        self.vecs_to_print              = None
        self.ds_handle                  = None
        self.caching                    = None
        self.use_s3                     = None
        self.s3_bucket                  = None
        self.use_compression            = None
        self.bigann                     = None
        self.indextype                  = None
        self.channel                    = None
        self.nprA                       = None
        self.nprB                       = None        
        self.efsA                       = None
        self.efsB                       = None
        self.faiss_threads              = None


        # AttributeSet
        self.full_afname                = None      # Columner
        self.full_std_afname            = None      # Columnar
        self.attribute_energies         = None
        self.aset                       = None
        self.quant_attr_data            = None      # Row-wise        
        
        # TransformedDataSet
        self.full_tf_fname              = None
        self.tf_handle                  = None

        # VAQIndex
        self.full_vaq_fname             = None 
        self.bq_fname                   = None  
        self.energies                   = None
        self.cset                       = None
        self.vaqdata                    = None
        self.bqdata                     = None  
        
        # Faiss Index
        self.faiss_index                = None        
        
        # QuerySet
        self.full_querydata_fname       = None
        self.Q                          = None # Query set placeholder
        self.Q_raw                      = None
        self.QQ                         = None # Quantized query set placeholder
        self.BQQ                        = None # Binary Quantized query set placeholder     
        self.q                          = None
        self.qq                         = None
        self.bqq                        = None 
        self.qset                       = None       
        self.num_queries                = None 
        self.attr_masks                 = None   
        self.attr_matching_cells        = None 
        self.predicate_sets             = None 
        self.filters                    = None
        self.candidate_count            = None 
        self.binary_quantization        = None 
        self.gt_data                    = None  
        self.gt_raw_data                = None
        self.gt_raw_dists_data          = None
        self.gt_attr_data               = None
        self.gt_attr_dists_data         = None
        self.gt_total_hits              = None
        self.query_inds                 = None   
        
        # Partitioner                  
        self.partition_vectors          = None
        self.partition_ids              = None
        self.partition_pops             = None
        self.partition_centroids        = None
        
        # Other
        self.warm_start                 = False
        self.efs_bytes_read             = None
        self.s3_gets                    = None
        self.qlbl_sizes                 = None
        self.qlbl_indices               = None
        self.qlbl_indptr                = None
        self.fbp_num_elements           = None
        self.fbp_mask                   = None
        self.fbp_offsets                = None
        self.s3_client                  = None
        self.cand_array                 = None                

        np.set_printoptions(linewidth=200)
        
        self.initialize()
        
    #----------------------------------------------------------------------------------------------------------------------------------------
    def initialize(self):
        
        # Imports done here to avoid circular references
        from sqlayer import GQA
        global gqa 
        gqa = GQA.getInstance()
        from sqlayer import GQP
        global gqp 
        gqp = GQP.getInstance()
       
        # Print floating-point numbers using a fixed point notation
        np.set_printoptions(suppress=True) 
        
        # Unload params
        self.unload_params()
        if self.mode == 'A':
            self.s3_client = gqa.S3_CLIENT
        elif self.mode == 'Q':
            self.s3_client = gqa.S3_CLIENT
            
        # Qsession
        # --------
        # Add derived parameters used in > one class
        self.total_file_words = self.num_vectors * (self.num_dimensions + 1)
        self.num_words_per_block = int(self.total_file_words / self.num_blocks)
        self.num_vectors_per_block = int(self.num_words_per_block / (self.num_dimensions + 1))
        self.tf_num_words_per_block = 0
        self.tf_num_vectors_per_block = 0
        self.tp_num_words_per_block = 0
        
        self.dim_means                  = None
        self.tf_dim_means               = None
        self.at_means                   = None
        self.tf_stdevs                  = None
        self.at_stdevs                  = None
        self.cov_matrix                 = None
        self.at_cov_matrix              = None
        self.transform_matrix           = None
        self.at_transform_matrix        = None
        self.cells                      = None
        self.attribute_cells            = None
        self.boundary_vals              = None    
        self.attribute_boundary_vals    = None
        self.sdc_lookup_lower           = None 
        self.sdc_lookup_upper           = None
        
        # Basic validations
        assert os.path.isdir(self.path), "Path entered : " + self.path + " does not exist!"
        assert (self.total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."
        assert self.mode in ('A','Q','P'), "Mode must be one of (A)llocator, (Q)uery or (P)rint"
        assert self.num_vectors > 0, "num_vectors must be greater than 0"
        assert self.num_dimensions > 0, "num_dimensions must be greater than 0"
        assert self.num_attributes >= 0, "num_attributes cannot be negative"
        assert self.num_blocks > 0, "num_blocks must be greater than 0"
        assert self.word_size > 0, "word_size must be greater than 0"
        assert self.bit_budget > 0, "bit_budget must be greater than 0"
        if self.mode == 'P' and self.vecs_to_print is None:
            print('Mode P requires a list of Vector IDs to be provided!')
            exit(1)
        if self.precision == 'approx_bqlb':
            self.binary_quantization = True
        else:
            self.binary_quantization = False
            
        if self.use_compression and self.channel == 'F':
            print('Channel F requires use_compression to be set to False!')
            exit(1)            

        # Partitioner
        # -----------
        self.partition_vectors      = None
        self.partition_ids          = None
        self.partition_pops         = None
        self.partition_centroids    = None
            
        # AttributeSet
        # ------------
        self.full_afname = os.path.join(self.path, '') + self.fname + '.af'
        self.full_std_afname = os.path.join(self.path, '') + self.fname + '.afstd'

        # DataSet
        # ------------
        self.full_fname = os.path.join(self.path, '') + self.fname
        
        # TransformedDataSet
        # ------------------
        self.full_tf_fname = os.path.join(self.path, '') + self.fname + '.tf'
        total_tf_file_words = (self.num_vectors * (self.num_dimensions))        
        assert (total_tf_file_words % self.num_blocks == 0) and (total_tf_file_words // self.num_blocks) % self.num_dimensions == 0, "Incorrect number of blocks specified"
        self.tf_num_words_per_block = int(total_tf_file_words / self.num_blocks)
        self.tf_num_vectors_per_block = int(self.tf_num_words_per_block / (self.num_dimensions))
        self.tp_num_words_per_block = int(total_tf_file_words / self.num_dimensions)

        # VAQIndex
        # --------
        self.full_vaq_fname = os.path.join(self.path, '') + self.fname + '.vaq'
        self.bq_fname = os.path.join(self.path, '') + self.fname + '.bq'   
        self.hammings           = None
        self.candidate_count    = 0
        self.candidates         = None
        self.candidate_hammings = None
            
        # QuerySet
        # --------
        # NB Path for the following files depend on whether num_allocators > 1 (also whether dmg belongs to QA or QP)
        if (self.num_allocators == 1) or (self.mode == 'Q'):
            qpath = self.path
        else:
            qpath = os.path.join(self.path, self.allocators_root, str(self.allocator_id))
        
        if self.query_fname is not None:
            self.full_querydata_fname = os.path.join(qpath, '') + self.query_fname
        else:
            self.full_querydata_fname = os.path.join(qpath, '') + self.fname + '_qry.npz'
            
        # Other
        # -----
        self.efs_bytes_read      = 0
        self.s3_gets             = 0
        self.fbp_num_elements    = np.ceil(self.num_vectors // 8)
        self.fbp_mask            = np.full((self.fbp_num_elements), 128, dtype=np.uint8)            
        self.fbp_offsets         = np.array([i * 8 for i in range(self.fbp_num_elements)],dtype=np.int32)               
        
    #----------------------------------------------------------------------------------------------------------------------------------------     
    def unload_params(self):
        self.exp_tag                 = self.params["exp_tag"]
        self.exp_run                 = int(self.params["exp_run"])
        self.path                    = self.params["path"]
        self.fname                   = self.params["fname"]
        self.mode                    = self.params["mode"]
        self.query_k                 = int(self.params["query_k"])  
        self.query_batchsize         = int(self.params["query_batchsize"])          
        self.query_fname             = self.params["query_fname"]  
        self.num_vectors             = int(self.params["num_vectors"])
        self.num_dimensions          = int(self.params["num_dimensions"])
        self.num_attributes          = int(self.params["num_attributes"])
        self.num_blocks              = int(self.params["num_blocks"])
        self.word_size               = int(self.params["word_size"])
        self.datatype                = self.str_to_dtype(self.params["datatype"])
        self.big_endian              = self.params["big_endian"]
        self.bit_budget              = int(self.params["bit_budget"])
        self.attribute_bit_budget    = int(self.params["attribute_bit_budget"])        
        self.precision               = self.params["precision"]
        self.inmem_vaqdata           = self.params["inmem_vaqdata"]
        self.adc_or_sdc              = self.params["adc_or_sdc"]
        self.bq_cutoff_perc          = int(self.params["bq_cutoff_perc"])
        self.check_recall            = self.params["check_recall"]
        self.num_allocators          = int(self.params["num_allocators"])
        self.allocators_root         = self.params["allocators_root"]
        self.allocator_id            = int(self.params["allocator_id"])
        self.fine_tune               = self.params["fine_tune"]
        self.centroid_factor         = np.float32(self.params["centroid_factor"])
        self.k_factor                = np.float32(self.params["k_factor"])
        self.bfr                     = int(self.params["bfr"])   
        self.l_max                   = int(self.params["l_max"])
        self.level                   = int(self.params["level"])
        self.vecs_to_print           = np.array(self.params["vecs_to_print"],dtype=np.uint32)
        self.caching                 = self.params["caching"]
        self.use_s3                  = self.params["use_s3"]        
        self.s3_bucket               = self.params["s3_bucket"]
        self.use_compression         = self.params["use_compression"]               
        self.bigann                  = self.params["bigann"]
        self.indextype               = self.params["indextype"]
        self.channel                 = self.params["channel"]
        self.nprA                    = int(self.params["nprA"])
        self.nprB                    = int(self.params["nprB"])
        self.efsA                    = int(self.params["efsA"])
        self.efsB                    = int(self.params["efsB"])
        self.faiss_threads           = int(self.params["faiss_threads"])        
    #----------------------------------------------------------------------------------------------------------------------------------------     
    # QSession   
    def process_timer(self, metric, start_timer):
        # end_timer = timeit.default_timer()
        # duration = end_timer - start_timer
        # update_stats(metric, duration)
        pass
    #----------------------------------------------------------------------------------------------------------------------------------------
    # QSession       
    def debug_timer(self, function, reference_time, message, indent=0):
        tabs = ''
        if DataManager.DEBUG:
            for i in range(indent + 1):
                tabs += '\t'
            current_time = timeit.default_timer()
            # msg = function + ' -> ' + message
            msg = function.ljust(50,' ') + ' -> ' + message.ljust(50,' ')
            elapsed = tabs + str(current_time - reference_time)
            print("[TIMER] " , msg , "Elapsed: ", elapsed.ljust(20,' '))   
    #----------------------------------------------------------------------------------------------------------------------------------------            
    # QSession       
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False            
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # AttributeSet        CONVERT TO S3
    def read_vector_attributes(self, vector_id, type='raw'):
        if type == 'raw':
            fname = self.full_afname
        elif type == 'std':
            fname = self.full_std_afname
        elif type == 'qnt':
             return self.quant_attr_data[vector_id,:]
        else:
            pass              
        
        if type in ('raw','std'):
            # Requires num_attributes random reads as data is stored 'columnar' (attribute by attribute)
            attribs = np.zeros(self.num_attributes, dtype=np.float32)
            with open(fname, mode='rb') as f:
                for i in range(self.num_attributes):
                    offset = np.uint64( ( (i*self.num_vectors) + vector_id) * self.word_size )
                    f.seek(offset, os.SEEK_SET)
                    attribs[i] = np.fromfile(file=f, count=1, dtype=np.float32)
            self.efs_bytes_read += (self.num_attributes * self.word_size)                    
            return attribs
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # DataSet           
    def ds_open_file(self):
        self.ds_handle = open(self.full_fname, mode='rb')
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # DataSet  
    def ds_close_file(self):
        self.ds_handle.close()
        self.ds_handle = None
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # DataSet           USED IN PRVD AND CALC_RAW_DISTANCES: CONVERT TO S3
    def ds_random_read_vector(self, vector_id):  
        vector_words = (self.num_dimensions + 1) 
        # offset       = np.uint64(vector_id * vector_words * self.word_size)
        offset       = np.uint64(vector_id * vector_words * self.datatype().itemsize)
        
        if self.ds_handle is None:
            with open(self.full_fname, mode='rb') as f:
                f.seek(offset, os.SEEK_SET)
                vector = np.fromfile(file=f, count=vector_words, dtype=self.datatype)                
        else:
            self.ds_handle.seek(offset, os.SEEK_SET)
            vector = np.fromfile(file=self.ds_handle, count=vector_words, dtype=self.datatype)

        self.efs_bytes_read += (vector_words * self.word_size)

        if vector.size > 0:
            vector = np.delete(vector, 0)
        return vector    
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # TransformedDataSet
    def ts_open_file(self):
        self.tf_handle = open(self.full_tf_fname)
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # TransformedDataSet
    def ts_close_file(self):
        self.tf_handle.close()
        self.tf_handle = None
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # TransformedDataSet            USED IN PRVD AND PHASE 2: CONVERT TO S3
    def tf_random_read(self, start_offset, num_words_random_read):  
        if self.tf_handle is None:
            with open(self.full_tf_fname, mode='rb') as f:
                f.seek(start_offset, os.SEEK_SET)
                block = np.fromfile(file=f, count=num_words_random_read, dtype=np.float32)
        else:        
            self.tf_handle.seek(start_offset, os.SEEK_SET)
            block = np.fromfile(file=self.tf_handle, count=num_words_random_read, dtype=np.float32)                 

        self.efs_bytes_read += (num_words_random_read * self.word_size)
                 
        if block.size > 0:
            block = np.reshape(block, (1, self.num_dimensions))  # Done this way round, rather than MATLAB [DIMENSION, 1]'.
        return block
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # VAQIndex                      USED IN QP: CONVERT TO S3
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)     
    def generate_vaq_block(self):
        dt              = DataManager.BITWISE_CONTAINER_DATATYPE
        dt_bits         = dt().itemsize * 8 
        num_vecs        = self.num_vectors
        bit_allocs      = np.uint8(np.ceil(np.log2(self.cells)))
        # segment         = np.zeros(num_vecs,dtype=dt)
        segment_residue = np.zeros(num_vecs,dtype=dt)
        seg_used        = np.uint8(0)
        residual_bits   = np.uint8(0)
        dim             = 0
        block_idx       = 0

        with open(self.full_vaq_fname, mode="rb") as f:

            while dim < self.num_dimensions:
                # f.seek(self.tp_num_words_per_block * block_idx, os.SEEK_SET)
                f.seek((self.tp_num_words_per_block * dt().itemsize * block_idx), os.SEEK_SET)
                block = np.fromfile(file=f, count=self.tp_num_words_per_block, dtype=dt)
                if block.size == 0:
                    break
                
                while dim < self.num_dimensions:
                    
                    if seg_used == dt_bits:
                        if residual_bits == 0:
                            seg_used = 0
                            segment_residue = np.zeros(num_vecs,dtype=dt)
                            block_idx += 1
                            break           # Next block
                        else:
                            out_block = np.bitwise_or(segment_residue, np.right_shift(block,dt_bits - (bit_allocs[dim] - residual_bits)))
                            # print()
                            # print(out_block)
                            seg_used = bit_allocs[dim] - residual_bits
                            dim += 1
                            residual_bits = 0
                            if self.candidate_count > 0:
                                yield out_block[self.candidates].reshape(self.candidate_count)
                            else:
                                yield out_block.reshape(self.num_vectors)
                    
                    elif seg_used == 0:
                        out_block = np.right_shift(block,dt_bits - bit_allocs[dim])
                        # print(np.binary_repr(out_block[0],dt_bits))
                        # print(out_block)                
                        seg_used += bit_allocs[dim]
                        dim += 1
                        if self.candidate_count > 0:
                            yield out_block[self.candidates].reshape(self.candidate_count)
                        else:
                            yield out_block.reshape(self.num_vectors)
                        
                    elif (seg_used > 0) and (bit_allocs[dim] <= (dt_bits - seg_used)):
                        out_block = np.right_shift(np.left_shift(block,seg_used),dt_bits - bit_allocs[dim])
                        # print(np.binary_repr(out_block[0],dt_bits))
                        # print(out_block)                
                        seg_used += bit_allocs[dim]
                        dim += 1
                        if self.candidate_count > 0:
                            yield out_block[self.candidates].reshape(self.candidate_count)
                        else:
                            yield out_block.reshape(self.num_vectors)

                    elif (seg_used > 0) and (bit_allocs[dim] > (dt_bits - seg_used)):                
                        segment_residue = np.right_shift(np.left_shift(block,seg_used), dt_bits - bit_allocs[dim])
                        residual_bits = dt_bits - seg_used
                        seg_used += residual_bits
                        block_idx += 1
                        break            # Next block     
    
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # VAQIndex              
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)
    def generate_vaq_block_mem(self):
        dt              = DataManager.BITWISE_CONTAINER_DATATYPE
        dt_bits         = dt().itemsize * 8 
        num_vecs        = self.num_vectors
        bit_allocs      = np.uint8(np.ceil(np.log2(self.cells)))
        # segment         = np.zeros(num_vecs,dtype=dt)
        segment_residue = np.zeros(num_vecs,dtype=dt)
        seg_used        = np.uint8(0)
        residual_bits   = np.uint8(0)
        dim             = 0
        block_idx       = 0

        while dim < self.num_dimensions:
            if self.candidate_count > 0:
                block = self.vaqdata[self.candidates,block_idx].reshape(self.candidate_count)
            else:
                block = self.vaqdata[:,block_idx].reshape(self.num_vectors)
            while dim < self.num_dimensions:
                
                if seg_used == dt_bits:
                    if residual_bits == 0:
                        seg_used = 0
                        segment_residue = np.zeros(num_vecs,dtype=dt)
                        block_idx += 1
                        break           # Next block
                    else:
                        out_block = np.bitwise_or(segment_residue, np.right_shift(block,dt_bits - (bit_allocs[dim] - residual_bits)))
                        # print()
                        # print(out_block)
                        seg_used = bit_allocs[dim] - residual_bits
                        dim += 1
                        residual_bits = 0
                        # if self.candidate_count > 0:
                        #     yield out_block[self.candidates].reshape(self.candidate_count)
                        # else:
                        #     yield out_block.reshape(self.num_vectors)
                        yield out_block
                
                elif seg_used == 0:
                    out_block = np.right_shift(block,dt_bits - bit_allocs[dim])
                    # print(np.binary_repr(out_block[0],dt_bits))
                    # print(out_block)                
                    seg_used += bit_allocs[dim]
                    dim += 1
                    # if self.candidate_count > 0:
                    #     yield out_block[self.candidates].reshape(self.candidate_count)
                    # else:
                    #     yield out_block.reshape(self.num_vectors)
                    yield out_block
                    
                elif (seg_used > 0) and (bit_allocs[dim] <= (dt_bits - seg_used)):
                    out_block = np.right_shift(np.left_shift(block,seg_used),dt_bits - bit_allocs[dim])
                    # print(np.binary_repr(out_block[0],dt_bits))
                    # print(out_block)                
                    seg_used += bit_allocs[dim]
                    dim += 1
                    # if self.candidate_count > 0:
                    #     yield out_block[self.candidates].reshape(self.candidate_count)
                    # else:
                    #     yield out_block.reshape(self.num_vectors)
                    yield out_block
                    
                elif (seg_used > 0) and (bit_allocs[dim] > (dt_bits - seg_used)):                
                    segment_residue = np.right_shift(np.left_shift(block,seg_used), dt_bits - bit_allocs[dim])
                    residual_bits = dt_bits - seg_used
                    seg_used += residual_bits
                    block_idx += 1
                    break            # Next block
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # VAQIndex                  USED IN PRVD: CONVERT TO S3
    # Read one vector from VAQIndex (Enhanced for bitwise processing)
    def read_vaq_vector(self, vec_id=None):

        dim_bitallocs   = np.uint8(np.log2(self.cells))
        elements        = []    
        bitsequence     = ""
        dt              = DataManager.BITWISE_CONTAINER_DATATYPE
        dt_wordsize     = dt().itemsize
        dt_size         = dt_wordsize * 8


        # If inmem_vaqdata, read from array and extract dimension values from container variables
        if self.inmem_vaqdata in ('inmem_oneshot', 'inmem_columnar'):
            for val in self.vaqdata[vec_id,:]:
                bitsequence += np.binary_repr(val,dt_size)

            startpos = np.uint64(0)
            for bits in dim_bitallocs:
                elements.append(np.uint8(int(bitsequence[startpos:startpos+bits],2))) 
                startpos += bits
            return np.asarray(elements, dtype=np.uint8)
        
        # Otherwise read from disk (needs a loop as stored by dimension, not vector. Also need to extract dims from container variables)
        num_vecs = self.num_vectors
        containers_per_vec = np.uint16(np.divide(dim_bitallocs.sum(), dt_size))
        
        with open(self.full_vaq_fname, mode="rb") as f:
            
            for container_no in range(containers_per_vec):
                offset = np.uint64((container_no * num_vecs * dt_wordsize) + (vec_id * dt_wordsize))
                f.seek(offset, os.SEEK_SET)
                container = np.fromfile(file=f, count=1, dtype=dt)
                bitsequence += np.binary_repr(container[0],dt_size)
                
        startpos = np.uint64(0)
        for bits in dim_bitallocs:
            elements.append(np.uint8(int(bitsequence[startpos:startpos+bits],2))) 
            startpos += bits
        return np.asarray(elements, dtype=np.uint8)        
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    # VAQIndex    
    # Print details of one or more vectors
    def prvd(self):
        
        bitallocs = np.uint8(np.log2(self.cells))

        # Loop over requested vectors
        for vec_id in self.vecs_to_print:
            
            print("Details for Vector ID : ", vec_id)
            print()
            
            if self.num_attributes > 0:
                print('ATTRIBUTES')

                # Read vector attributes (raw and standardized) from AttributeSet
                atraw = self.read_vector_attributes(vec_id,'raw')
                atstd = self.read_vector_attributes(vec_id,'std')
                atqnt = self.read_vector_attributes(vec_id,'qnt')
                
                print('{:^4s}     {:^10s} {:^10s} {:^4s}    {:<10s}   {:<10s}  {:<6s}  {:>20s} '.format('Attr', 'raw', 'std', 'qnt', 'Boundary', 'LB', 'UB', 'Surrounding Boundaries'))
                # Loop over Attributes            
                for att_no in range(self.num_attributes):
                    boundary_start = (atqnt[att_no] - 4) if (atqnt[att_no] - 4) > 0 else 0
                    boundary_stop  = (atqnt[att_no] + 5) if (atqnt[att_no] + 5) <= self.attribute_cells[att_no] else self.attribute_cells[att_no] + 1
                    B1 = np.abs(np.subtract(atstd[att_no], self.attribute_boundary_vals[atqnt[att_no], att_no]))
                    B2 = np.abs(np.subtract(atstd[att_no], self.attribute_boundary_vals[atqnt[att_no]+1, att_no]))
                    LB = min(B1,B2)
                    UB = max(B1,B2)

                    print('{:^4d}  {:>10.4f} {:>10.4f}   {:>4d} {:>10.4f} {:>10.4f} {:>10.4f}    '.format(att_no, atraw[att_no], atstd[att_no], atqnt[att_no], self.attribute_boundary_vals[atqnt[att_no], att_no], LB, UB), end='')
                    print(self.attribute_boundary_vals[boundary_start:boundary_stop, att_no])            

            # Read DataSet vector
            rvec = self.ds_random_read_vector(vec_id)
                                
            # Read vector from TransformedDataSet
            start_offset = np.uint64(vec_id * self.num_dimensions * self.word_size)
            tvec = self.tf_random_read(start_offset, self.num_dimensions) 
                        
            # Read quantised vector from VAQIndex
            qvec = self.read_vaq_vector(vec_id)
            
            print()
            print('DIMENSIONS')
            # print('{:^4s} {:^4s} {:^10s} {:^4s}          {:<10s}         {:<10s} {:<6s} {:>20s} '.format('Dim', 'bits', 'tvec', 'qvec', 'Boundary', 'LB', 'UB', 'Surrounding Boundaries'))
            print('{:^4s} {:^4s}  {:^10s} {:^10s}{:^4s}     {:<10s}     {:<10s} {:<6s} {:>20s} '.format('Dim', 'bits', 'raw', 'tvec', 'qvec', 'Boundary', 'LB', 'UB', 'Surrounding Boundaries'))            
            # Loop over Dimensions            
            for dim_no in range(self.num_dimensions):
                # boundary_start = (qvec[dim_no] - 4) if (qvec[dim_no] - 4) > 0 else 0
                boundary_start = (qvec[dim_no] - 4) if qvec[dim_no] > 4 else 0
                boundary_stop  = (qvec[dim_no] + 5) if (qvec[dim_no] + 5) <= self.num_dimensions else self.num_dimensions + 1
                B1 = np.abs(np.subtract(tvec[0,dim_no], self.boundary_vals[qvec[dim_no], dim_no]))
                B2 = np.abs(np.subtract(tvec[0,dim_no], self.boundary_vals[qvec[dim_no]+1, dim_no]))
                LB = min(B1,B2)
                UB = max(B1,B2)
                # print('{:^4d} {:>10.4f} {:>4d}    {:>10.4f}      '.format(dim_no, tvec[0,dim_no], qvec[dim_no], self.boundary_vals[qvec[dim_no], dim_no]), end='')
                
                # print('{:^4d} {:^4d}{:>10.4f} {:>4d}         {:>10.4f}     {:>10.4f} {:>10.4f}    '.format(dim_no, bitallocs[dim_no], tvec[0,dim_no], qvec[dim_no], self.boundary_vals[qvec[dim_no], dim_no], LB, UB), end='')
                print('{:^4d} {:^4d}{:>10.4f}{:>10.4f} {:>4d}    {:>10.4f} {:>10.4f} {:>10.4f}    '.format(dim_no, bitallocs[dim_no], rvec[dim_no], tvec[0,dim_no], qvec[dim_no], self.boundary_vals[qvec[dim_no], dim_no], LB, UB), end='')                
                print(self.boundary_vals[boundary_start:boundary_stop, dim_no])
            
            print()
    #----------------------------------------------------------------------------------------------------------------------------------------    
    # VAQIndex        
    # Compare two floats for almost-equalness
    def _fleq(self, fl1, fl2):
        rtol = 1e-05
        atol = 1e-08
        return np.isclose(fl1, fl2, rtol, atol, True)
    #----------------------------------------------------------------------------------------------------------------------------------------    
    # QuerySet                      
    def open_querydata_file(self):

        if self.use_s3:
            qd_fname = self.s3_download(ftype='qry')
        else:
            qd_fname = self.full_querydata_fname
            self.efs_bytes_read += os.path.getsize(qd_fname)
        with np.load(qd_fname) as data_qry:
                        queries                     = data_qry['QUERIES']                            
                        self.gt_raw_data            = data_qry['GT_RAW']   
                        self.gt_raw_dists_data      = data_qry['GT_RAW_DISTS']   
                        self.gt_attr_data           = data_qry['GT_ATTR']   
                        self.gt_attr_dists_data     = data_qry['GT_ATTR_DISTS']
                        self.query_inds             = data_qry['INDS']
                        if not self.bigann:
                            self.predicate_sets         = data_qry['PRED_SETS']
                        else:
                            self.qlbl_sizes             = data_qry['QLBL_SIZES']          
                            self.qlbl_indices           = data_qry['QLBL_INDICES']
                            self.qlbl_indptr            = data_qry['QLBL_INDPTR']                        

        # Reshape and trim identifiers
        print("queries shape before reshape + trim identifiers: ", np.shape(queries))
        queries = np.reshape(queries, (-1, self.num_dimensions+1), order="C")
        queries = np.delete(queries, 0, 1)
        print("queries shape after reshape + trim identifiers: ", np.shape(queries))
        
        # Set up remaining variables
        self.Q = queries
        self.num_queries = np.shape(queries)[0]
        # self.first_stage = np.zeros((self.num_queries), dtype=np.uint32)
        # self.second_stage = np.zeros((self.num_queries), dtype=np.uint32)     
        
        if (self.num_attributes > 0) and (not self.bigann):
            self.calculate_attribute_masks()    
            
        if self.check_recall:
            self.gt_total_hits = 0
            if self.num_attributes > 0:
                self.gt_data = self.gt_attr_data
            else:
                self.gt_data = self.gt_raw_data
    
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    # QuerySet                  
    def calculate_attribute_masks(self):      
        # self.attr_matching_cells = np.ones((self.num_queries, np.max(self.attribute_cells)+1, self.num_attributes), dtype=np.uint8)
        self.attr_matching_cells = np.ones((self.num_queries, np.max(self.attribute_cells.astype(np.uint16))+1, self.num_attributes), dtype=np.uint8)   # Cast required at numpy 2.0
        self.filters = np.zeros((self.num_queries, self.num_attributes), dtype=np.uint8)

        for query_num, query_pred_set in enumerate(self.predicate_sets):
            for i in range(0, len(query_pred_set), 4):
                pred = query_pred_set[i:i+4]
                attr_id = int(pred[0])
                operator = pred[1]
                if operator != '':
                    self.filters[query_num, attr_id] = 1
                    raw_operand1 = np.float32(pred[2])
                    operand1 = np.divide(np.subtract(raw_operand1,self.at_means[:,attr_id]), self.at_stdevs[:,attr_id])
                    # operand1 = np.dot(np.subtract(raw_operand1,self.at_means[:,attr_id]), self.at_transform_matrix[attr_id,attr_id])
                    # operand1 = raw_operand1
                    
                    if len(pred[3]) > 0:
                        raw_operand2 = np.float32(pred[3])
                        operand2 = np.divide(np.subtract(raw_operand2, self.at_means[:,attr_id]), self.at_stdevs[:,attr_id])
                        # operand2 = np.dot(np.subtract(raw_operand2,self.at_means[:,attr_id]), self.at_transform_matrix[attr_id,attr_id])                         
                        # operand2 = raw_operand2
                    else:
                        raw_operand2 = None
                        operand2 = None

                    if operator == '<': 
                        self.attr_matching_cells[query_num, :,attr_id] = np.where(self.attribute_boundary_vals[:,attr_id] < operand1, 1, 0)
                        
                    elif operator == '<=':
                        self.attr_matching_cells[query_num, :,attr_id] = np.where(self.attribute_boundary_vals[:,attr_id] <= operand1, 1, 0)

                    if operator == '=': 
                        cond = self.attribute_boundary_vals[:,attr_id] <= operand1
                        left = np.max(self.attribute_boundary_vals[:,attr_id][cond])
                        condition1 = (self.attribute_boundary_vals[:,attr_id] == left)
                        self.attr_matching_cells[query_num, :,attr_id] = np.where(condition1, 1, 0)

                    elif operator in ('>', '>='):
                        cond = self.attribute_boundary_vals[:,attr_id] <= operand1
                        left = np.max(self.attribute_boundary_vals[:,attr_id][cond])
                        condition1 = (self.attribute_boundary_vals[:,attr_id] == left)
                        self.attr_matching_cells[query_num, :,attr_id] = np.where(condition1 | (self.attribute_boundary_vals[:,attr_id] > operand1), 1, 0)

                    elif operator == 'B': # > op1, <= op2
                        # This is necessary when using klt
                        if operand1 > operand2:
                            operand1, operand2 = operand2, operand1
                        
                        cond = self.attribute_boundary_vals[:,attr_id] <= operand1
                        left = np.max(self.attribute_boundary_vals[:,attr_id][cond])
                        cond = self.attribute_boundary_vals[:,attr_id] >= operand2 
                        right = np.min(self.attribute_boundary_vals[:,attr_id][cond])
                        condition1 = (self.attribute_boundary_vals[:,attr_id] == left)
                        condition2 = (self.attribute_boundary_vals[:,attr_id] == right)
                        condition3 = ((self.attribute_boundary_vals[:,attr_id] > left) & (self.attribute_boundary_vals[:,attr_id] < right))
                        self.attr_matching_cells[query_num, :,attr_id] = np.where(condition1 | condition2 | condition3, 1, 0)

                    dummy=0        
    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # QuerySet    
    def apply_attribute_filters(self, query_idx):
        attr_reft = timeit.default_timer()
        
        if self.num_attributes > 0:
            self.attr_masks = np.packbits(np.ones((self.num_vectors), dtype=np.uint8), axis=0)

            # Iterate through attributes
            for i in range(self.num_attributes):
                if self.filters[query_idx,i] == 0:
                    continue
                # Index into self_attr_matching_cells using self.quant_attr_data
                matching_rows = np.packbits(self.attr_matching_cells[query_idx, self.quant_attr_data[:, i], i],axis=0)
                self.attr_masks = np.bitwise_and(self.attr_masks, matching_rows)

            # Having iterated through attributes, we have a packbits mask for all vectors. Unpack, we need idx positions
            # self.candidates = np.array(bitarray(endian='big', buffer=self.attr_masks).search(1),dtype=np.uint32)
            # self.candidate_count = self.candidates.shape[0]
            
            # # bitarray version
            # self.candidate_count = np.array(bitarray(endian='big', buffer=self.attr_masks).count(1),dtype=np.uint32)
            
            # NumPy version
            self.candidate_count = np.array((np.bitwise_count(self.attr_masks).sum()), dtype=np.uint32)
            
            # self.attr_filtering_count = np.subtract(self.num_vectors, self.candidate_count)
            print('Candidates after attribute filtering : ', self.candidate_count)
        else:
            self.candidate_count = self.num_vectors
            print('No attribute filtering - Candidates unchanged : ', self.candidate_count)
            
        msg = 'Query : ' + str(query_idx) + ' Attribute Filtering Duration'
        self.debug_timer('DataManager.apply_attribute_filters',attr_reft, msg)          
    # ---------------------------------------------------------------------------------------------------------------------------------------- 
    # Utility
    def find_bit_positions(self, ba):
        # bacopy      = ba.copy()
        bacopy      = ba.copy()[0:self.fbp_offsets.shape[0]]
        results     = np.zeros((8,self.fbp_num_elements),dtype=np.int32)
        for shift in range(8):
            
            # matches         = np.int32(np.right_shift(np.bitwise_and(bacopy, self.fbp_mask),7))
            # results[shift]  = np.where(matches==1, self.fbp_offsets + shift,-1)
            matches         = np.int32(np.bitwise_and(bacopy, self.fbp_mask))
            results[shift]  = np.where(matches==128, self.fbp_offsets + shift,-1)             
            
            bacopy          = np.left_shift(bacopy,1)
            shift += 1

        out = np.ravel(results, order='F')
        return np.uint32(out[out>=0])    
    # ----------------------------------------------------------------------------------------------------------------------------------------     
    # Labels    
    def apply_label_filters(self, query_idx):
        attr_reft = timeit.default_timer()    
        
        # Initialize
        attr_masks      = np.zeros((self.num_vectors), dtype=np.uint8)
        self.candidate_count = 0

        # Get query label indices
        qwords = self.qlbl_indices[self.qlbl_indptr[query_idx] : self.qlbl_indptr[query_idx + 1]]
        assert qwords.size in (1, 2), "Wrong number of qwords found : " + str(qwords.size)
        w1 = qwords[0]
        if qwords.size == 2:
            w2 = qwords[1]
        else:
            w2 = -1    
        
        # Find vectors with matching labels
        matching_vecs1 = self.lbl_csrt_indices[self.lbl_csrt_indptr[w1] : self.lbl_csrt_indptr[w1 + 1]]
        if w2 != -1:
            matching_vecs2 = self.lbl_csrt_indices[self.lbl_csrt_indptr[w2] : self.lbl_csrt_indptr[w2 + 1]]
            candidates = np.intersect1d(matching_vecs1, matching_vecs2)        
        else:
            candidates = matching_vecs1

        # Update mask
        attr_masks[candidates] = 1
        self.attr_masks = np.packbits(attr_masks, axis=0)
        self.candidate_count = candidates.shape[0]

        msg = 'Query : ' + str(query_idx) + ' Label Filtering Duration'
        self.debug_timer('DataManager.apply_label_filters',attr_reft, msg)          
    # ----------------------------------------------------------------------------------------------------------------------------------------            
    # QuerySet
    def transform_queries(self):
        X = self.Q
        Y = np.subtract(X, self.dim_means)        
        Z = np.matmul(Y,self.transform_matrix)
        self.Q = Z
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # QuerySet
    # Quantize the transformed queries
    def quantize_queries(self):
        self.QQ = np.zeros_like(self.Q, dtype=np.uint8)
        # Loop over dimensions
        for dim in range(self.num_dimensions):
            block = self.Q[:,dim]
            self.qset = np.full(self.Q.shape[0], -1, dtype=np.int16)   
            for i in range(self.cells[dim]):
                l = self.boundary_vals[i, dim]
                r = self.boundary_vals[i + 1, dim]
                A = np.where(np.logical_and(block >= l, block < r))[0]
                self.qset[A] = i

            # Deal with values above max threshold for dimension
            # unallocated = np.where(self.qset < 0)[0]
            # self.qset[unallocated] = self.cells[dim]
            
            unallocated = np.where(self.qset < 0)[0]
            if unallocated.shape[0] > 0:
                print('Query Dimension ', dim, ' has value ', block, ' outside boundary val range for this partition : ', np.min(self.boundary_vals[:,dim]), ' : ', np.max(self.boundary_vals[:,dim]) ,flush=True)
                self.qset[unallocated] = self.cells[dim] - 1            
            
            
            self.QQ[:,dim] = np.uint8(self.qset)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    # Returns a np.packbits array (sized to match the number of vectors in the partition), plus a count of the number of attribute-matching partition vectors
    def get_filtered_partition_vectors(self, partition_id):
        # candidates = np.unpackbits(np.bitwise_and(self.partition_vectors_packed[partition_id], self.attr_masks))
        # candidates = candidates[np.where(self.partition_vectors[:,partition_id] == 1)[0]]

        # 20th Sep
        candidates  = np.bitwise_and(self.partition_vectors[:,partition_id], self.attr_masks)
        pv_part     = self.partition_vectors[:,partition_id].copy()
        # selection   = np.array(bitarray(endian='big', buffer=pv_part).search(1),dtype=np.uint32)
        
        # # Using bitarray
        # selection   = np.array(list(bitarray(endian='big', buffer=pv_part).search(1)),dtype=np.uint32)
        
        # Using custom method
        selection  = self.find_bit_positions(pv_part)
        
        candidates  = np.unpackbits(candidates)[selection]
        
        return np.packbits(candidates), np.sum(candidates)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def unset_qa_globals(self):
        # TEMPORARY REQUIREMENT ONLY - WHEN RUNNING IN AWS GLOBALS WON'T BE VISIBLE ACROSS QUERYPROCESSORS
       
        global gqa
        
        gqa.path = gqa.fname = None
        gqa.partition_vectors = gqa.partition_ids = gqa.partition_pops = gqa.partition_centroids = None
        gqa.at_means = gqa.at_stdevs = gqa.attribute_cells = gqa.attribute_boundary_vals = None
        gqa.dim_means = gqa.cov_matrix = gqa.transform_matrix = None
        gqa.quant_attr_data = None    
        gqa.lbl_counters = gqa.lbl_csrt_indices = gqa.lbl_csrt_indptr = None        
            
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def unset_qp_globals(self):
        # TEMPORARY REQUIREMENT ONLY - WHEN RUNNING IN AWS GLOBALS WON'T BE VISIBLE ACROSS QUERYPROCESSORS
        
        global gqp
        
        gqp.path = gqp.fname = None
        gqp.dim_means = gqp.cov_matrix = gqp.transform_matrix = None
        gqp.tf_dim_means = gqp.tf_stdevs = None
        gqp.cells = gqp.boundary_vals = gqp.sdc_lookup_lower = gqp.sdc_lookup_upper = None
        gqp.vaqdata = gqp.bqdata = None
        gqp.indextype = gqp.faiss_index = None
            
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Utility
    def find_file_by_suffix(self, suffix):
        hit_count = 0
        hits = []
        for file in os.listdir(self.path):
            if file.endswith(suffix):
                hits.append(file)
                hit_count += 1
        if hit_count > 0:
            return True
        else:
            return False    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Utility
    def find_partition_vector(self, vecno):
        # Find packed vector container and remainder
        ans = np.divmod(vecno, 8)
        container   = ans[0]
        rem         = ans[1]
        # remexp      = 2 ** rem
        remexp      = 2 ** (8 - rem - 1)
                
        # Create mask
        mask = np.full_like(self.partition_vectors[container,:], remexp, dtype=np.uint8)
        
        # Apply mask and find max container
        partition = np.argmax(np.bitwise_and(self.partition_vectors[container,:], mask))
        
        offset = np.unpackbits(self.partition_vectors[0:container,partition]).sum() + np.unpackbits(self.partition_vectors[container,partition])[0:rem].sum()
        
        return partition, offset
    # ----------------------------------------------------------------------------------------------------------------------------------------  
    # Utility
    def find_partition_vecids(self, vecids, partid):
        lookup = np.unpackbits(self.partition_vectors[:,partid])
        return np.where(lookup == 1)[0][vecids]
    
        # nonzero_ids             = np.trim_zeros(vecids,'b')
        # converted_nonzero_ids   = np.uint32(np.where(lookup == 1)[0][nonzero_ids])
        # pad_width               = (0,self.query_k - len(converted_nonzero_ids))
        # # return np.pad(converted_nonzero_ids, pad_width, mode='constant', constant_values=np.iinfo(converted_nonzero_ids.dtype).max)  
        # return np.pad(converted_nonzero_ids, pad_width, mode='constant', constant_values=-1)     
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Utility
    def str_to_dtype(self, intype):
        types = {'int8' : np.int8,   'int16': np.int16,     'int32': np.int32,     'int64': np.int64,
                'uint8': np.uint8, 'uint16': np.uint16,   'uint32': np.uint32,   'uint64': np.uint64,
                                'float16': np.float16, 'float32': np.float32, 'float64': np.float64}
        return types[intype]
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Partitioner        
    def load_partitioner_vars(self):
        print("Loading partitioner variables from ", self.path)
        with np.load(os.path.join(self.path, '') + self.fname + '.ptnrvars.npz') as data:
            self.partition_vectors      = data['PART_VECTORS']
            self.partition_ids          = data['PART_IDS']
            self.partition_pops         = data['PART_POPS']
            self.partition_centroids    = data['PART_CENTROIDS']        
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    # DataManager
    def load_for_prvd(self):

        ptnrvars_present = self.find_file_by_suffix('ptnrvars.npz')
        vaqvars_present  = self.find_file_by_suffix('vaqvars.npz')
        if ptnrvars_present:
            self.load_qa_data()
        if vaqvars_present:
            self.load_qp_data()       
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def load_data(self):
        
        if self.mode == 'A':
            self.load_qa_data()
        elif self.mode == 'Q' and self.indextype == 'OSQ':
            self.load_qp_data()
        elif self.mode == 'Q' and self.indextype != 'OSQ':            
            self.load_qp_data_faiss()
        elif self.mode == 'P':
            pass
        else:
            print('DataManager : load_data received invalid Mode -> ', self.mode)
            exit(1)
            
            # self.load_vaqdata()
            # print('DataManager : Data Loaded!')        
            # print()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def s3_download(self, ftype=None):
        
        local_path  = Path('/tmp/')
        
        # Below now done in intialize
        # if self.mode == 'A':
        #     global gqa
        #     s3_client = gqa.S3_CLIENT
        # else:
        #     global gqp
        #     s3_client = gqp.S3_CLIENT

        path_for_s3 = self.path.replace("/mnt/squash/", "")
        print("In s3_download, path_for_s3: ", path_for_s3)
            
        if ftype == 'qa':
            # Need qavars.npz (dataset root folder) - self.path is the root directory for the dataset (eg datasets/sift1m/)
            # qavars_key  = os.path.join(self.path, '') + self.fname + '.qavars.npz'
            qavars_key  = os.path.join(path_for_s3, '') + self.fname + '.qavars.npz'
            # print("qavars_key: ", qavars_key)
            qavars_file = os.path.join(local_path,'') + self.fname + '.qavars.npz'
            self.s3_client.download_file(self.s3_bucket, qavars_key, qavars_file)
            self.s3_gets += 1             
            return qavars_file

        elif ftype == 'qry':
            # Need qry.npz (allocator folder) - self.path is the root directory for the dataset (eg datasets/sift1m/)
            # qry_key     = os.path.join(self.path, self.allocators_root, self.allocator_id, '') + self.fname + '_qry.npz'
            qry_key     = os.path.join(path_for_s3, str(self.allocators_root), str(self.allocator_id), '') + self.fname + '_qry.npz'
            qry_file    = os.path.join(local_path,'') + self.fname + '_qry.npz'    
            self.s3_client.download_file(self.s3_bucket, qry_key, qry_file)
            self.s3_gets += 1               
            return qry_file

        elif ftype == 'qp':
            qp_bucket = self.s3_bucket + '-' + os.path.split(self.path)[1]
            if self.indextype == 'OSQ':
                qpvars_key  = os.path.join(path_for_s3, '') + self.fname + '.qpvars.npz'
                qpvars_file = os.path.join(local_path,'') + self.fname + '.qpvars.npz'
            else:
                qpvars_key  = os.path.join(path_for_s3, '') + self.fname + '.qpvars_' + self.indextype + '.npz'
                qpvars_file = os.path.join(local_path,'') + self.fname + '.qpvars_' + self.indextype + '.npz'
            # self.s3_client.download_file(self.s3_bucket, qpvars_key, qpvars_file)
            self.s3_client.download_file(qp_bucket, qpvars_key, qpvars_file)
            self.s3_gets += 1               
            return qpvars_file
        
        else:
            'DataManager->s3_download: Invalid ftype ', ftype, ' requested!'
            return '1'

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def send_qp_candfile(self, pno, fkey, arr):
        
        if not self.use_s3:
            # Writing cands file to EFS/local
            tmp_path = os.path.join(self.path, 'tmp')
            if not os.path.exists(tmp_path):
                print("Failed to find file channel path : ",tmp_path)
                exit(1)                      
            cand_full_fname = os.path.join(tmp_path, '') + fkey
            np.savez(cand_full_fname, CANDS=arr)
                
        else:
            # Writing cands file to S3
            np.set_printoptions(threshold=np.inf)
            cand_array_str = str(zlib.compress(arr, level=DataManager.COMPRESSION_LEVEL))

            full_fkey = os.path.join(self.path, 'tmp', fkey)
            qp_bucket = self.s3_bucket + '-' + os.path.split(self.path)[1]
            # result = self.s3_client.put_object(Bucket=self.s3_bucket, Key=full_fkey, Body=cand_array_str)
            result = self.s3_client.put_object(Bucket=qp_bucket, Key=full_fkey, Body=cand_array_str)
            res = result.get('ResponseMetadata')
            status_code = res.get('HTTPStatusCode')

            if status_code != 200:
                print("[ERROR]: Failed to upload csr file to s3 -> Bucket : ", self.s3_bucket, " File Name : ", cand_full_fname, " ERROR CODE : ", str(status_code), flush=True)
                os._exit(1)	            
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def receive_qp_candfile(self, fkey, num_queries):
        
        root_path = os.path.split(os.path.split(self.path)[0])[0]
        if not self.use_s3:
            # Reading cands file from EFS/local
            tmp_path = os.path.join(root_path, 'tmp')
            cand_full_fname = os.path.join(tmp_path, '') + fkey + '.npz'
            print("Loading candidates from ", cand_full_fname)
            if os.access(cand_full_fname, os.R_OK):
                with np.load(cand_full_fname) as cands:
                    self.cand_array = cands['CANDS']
                return True
            else:
                return False

        else:
            # Reading cands file from S3
            np.set_printoptions(threshold=np.inf)
            full_fkey = os.path.join(root_path, 'tmp', fkey)
            qp_bucket = self.s3_bucket + '-' + os.path.split(self.path)[1]
            try:
                # Download object from s3
                # obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=full_fkey)
                obj = self.s3_client.get_object(Bucket=qp_bucket, Key=full_fkey)
                obj_str = obj["Body"].read().decode('utf-8') 
    
            except ClientError as e:
                # error_code = e.response["Error"]["Code"]
                # print("[ERROR]: Failed to download file from s3 -> Filename : ", full_fkey, " ERROR CODE : ", str(error_code), flush=True)
                return False

            decomp           = zlib.decompress(eval(obj_str))
            self.cand_array  = np.frombuffer(decomp, dtype=np.uint8).reshape(num_queries, self.fbp_num_elements)
            return True
       
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # DataManager
    def load_qa_data(self):
        
        global gqa        
        stub = os.path.join(self.path, '') + self.fname
        
        print()
        print('DataManager : Loading Data for Mode ', self.mode)
        print()
        
        if (gqa.path == self.path) and (gqa.fname == self.fname):
            print('Loading QA Data from Global Area!')
            self.warm_start                 = True
        else:
            print('Loading QA Data from File!')
            self.warm_start = False
            # gqa.reset()
            self.unset_qa_globals()
            gqa.path       = self.path
            gqa.fname      = self.fname
            if self.use_s3:
                qa_fname = self.s3_download(ftype='qa')
            else:
                qa_fname = stub + '.qavars.npz'
                self.efs_bytes_read += os.path.getsize(qa_fname)
            
            with np.load(qa_fname) as data_qa:
                gqa.partition_vectors          = data_qa['PART_VECTORS']
                gqa.partition_ids              = data_qa['PART_IDS']
                gqa.partition_pops             = data_qa['PART_POPS']
                gqa.partition_centroids        = data_qa['PART_CENTROIDS']
                gqa.dim_means                  = data_qa['DIM_MEANS']                            
                gqa.cov_matrix                 = data_qa['COV_MATRIX']                            
                gqa.transform_matrix           = data_qa['TRANSFORM_MATRIX']
                if not self.bigann:
                    gqa.quant_attr_data            = data_qa['QATT_DATA']                                                                         
                    gqa.at_means                   = data_qa['AT_MEANS']                            
                    gqa.at_stdevs                  = data_qa['AT_STDEVS']
                    gqa.attribute_cells            = data_qa['AT_CELLS']                            
                    gqa.attribute_boundary_vals    = data_qa['AT_BVALS']
                else:
                    gqa.lbl_counters               = data_qa['LBL_COUNTERS']
                    # gqa.lbl_vocab_sig              = data_qa['LBL_VOCAB_SIG']
                    # gqa.lbl_ds_sig                 = data_qa['LBL_DSSIG']
                    gqa.lbl_csrt_indices           = data_qa['LBL_INDICES']
                    gqa.lbl_csrt_indptr            = data_qa['LBL_INDPTR']
                
                
        # In both cases point instance variables to globals
        self.partition_vectors                  = gqa.partition_vectors
        self.partition_ids                      = gqa.partition_ids
        self.partition_pops                     = gqa.partition_pops
        self.partition_centroids                = gqa.partition_centroids
        self.dim_means                          = gqa.dim_means
        self.cov_matrix                         = gqa.cov_matrix
        self.transform_matrix                   = gqa.transform_matrix
        if not self.bigann:        
            self.at_means                           = gqa.at_means
            self.at_stdevs                          = gqa.at_stdevs
            self.attribute_cells                    = gqa.attribute_cells
            self.attribute_boundary_vals            = gqa.attribute_boundary_vals
            self.quant_attr_data                    = gqa.quant_attr_data
        else:
            self.lbl_counters                       = gqa.lbl_counters
            # self.lbl_vocab_sig                      = gqa.lbl_vocab_sig
            # self.lbl_ds_sig                         = gqa.lbl_ds_sig
            self.lbl_csrt_indices                   = gqa.lbl_csrt_indices
            self.lbl_csrt_indptr                    = gqa.lbl_csrt_indptr
                        
        # for pv in range(self.partition_vectors.shape[1]):
        #     self.partition_vectors_packed.append(np.packbits(self.partition_vectors[:,pv]))

        print('DataManager : QA Data Loaded!')        
        print()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def load_qp_data(self):
        
        global gqp
        stub = os.path.join(self.path, '') + self.fname
                
        print()
        print('DataManager : Loading Data for Mode ', self.mode)
        print()
        
        # if (gqp.path == self.path) and (gqp.fname == self.fname):
        if (gqp.path == self.path) and (gqp.fname == self.fname) and (gqp.indextype == self.indextype):        
            print('Loading QP Data from Global Area!')
            self.warm_start                 = True
        else:
            print('Loading QP Data from File!')
            self.warm_start = False
            # gqp.reset()
            self.unset_qp_globals()
            gqp.path       = self.path
            gqp.fname      = self.fname
            gqp.indextype  = self.indextype
            if self.use_s3:
                qp_fname = self.s3_download(ftype='qp')
            else:
                qp_fname = stub + '.qpvars.npz'
                self.efs_bytes_read += os.path.getsize(qp_fname)
            with np.load(qp_fname) as data_qp:
                # if not self.bigann:
                #     gqp.at_means                   = data_qp['AT_MEANS']                            
                #     gqp.at_stdevs                  = data_qp['AT_STDEVS']
                #     gqp.attribute_cells            = data_qp['AT_CELLS']                            
                #     gqp.attribute_boundary_vals    = data_qp['AT_BVALS']
                #     gqp.quant_attr_data            = data_qp['QATT_DATA']             
                gqp.dim_means                  = data_qp['DIM_MEANS']
                gqp.cov_matrix                 = data_qp['COV_MATRIX']
                gqp.transform_matrix           = data_qp['TRANSFORM_MATRIX']
                gqp.tf_dim_means               = data_qp['TF_DIM_MEANS']
                gqp.tf_stdevs                  = data_qp['TF_STDEVS']                            
                gqp.cells                      = data_qp['CELLS']
                gqp.boundary_vals              = data_qp['BOUNDARY_VALS']
                gqp.sdc_lookup_lower           = data_qp['SDC_LOOKUP_LOWER']                            
                gqp.sdc_lookup_upper           = data_qp['SDC_LOOKUP_UPPER']
                   
                
                if self.inmem_vaqdata in ('inmem_oneshot','inmem_columnar'):                        
                    gqp.vaqdata        = data_qp['VAQ_DATA']
                    if self.mode != 'P':
                        if self.inmem_vaqdata == 'inmem_oneshot':
                            msg = '(ONESHOT) IN-MEMORY VAQ PROCESSING SELECTED!'
                        elif self.inmem_vaqdata == 'inmem_columnar':
                            msg = '(COLUMNAR) IN-MEMORY VAQ PROCESSING SELECTED!'
                        print(msg)
                        print()                    
                
                if self.binary_quantization:
                    gqp.bqdata         = data_qp['BQ_DATA']
                
        # In both cases point instance variables to globals
        # if not self.bigann:
        #     self.at_means                   = gqp.at_means
        #     self.at_stdevs                  = gqp.at_stdevs
        #     self.attribute_cells            = gqp.attribute_cells
        #     self.attribute_boundary_vals    = gqp.attribute_boundary_vals
        #     self.quant_attr_data            = gqp.quant_attr_data        
        self.dim_means                  = gqp.dim_means
        self.cov_matrix                 = gqp.cov_matrix
        self.transform_matrix           = gqp.transform_matrix
        self.tf_dim_means               = gqp.tf_dim_means
        self.tf_stdevs                  = gqp.tf_stdevs
        self.cells                      = gqp.cells
        self.boundary_vals              = gqp.boundary_vals
        self.sdc_lookup_lower           = gqp.sdc_lookup_lower    
        self.sdc_lookup_upper           = gqp.sdc_lookup_upper
        self.vaqdata                    = gqp.vaqdata
        self.bqdata                     = gqp.bqdata
        
        
        print('DataManager : QP Data Loaded!')        
        print()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # DataManager
    def load_qp_data_faiss(self):
        
        global gqp
        stub = os.path.join(self.path, '') + self.fname
        
        print()
        print('DataManager : Loading (Faiss) Data for Mode ', self.mode)
        print()
        
        if (gqp.path == self.path) and (gqp.fname == self.fname) and (gqp.indextype == self.indextype):
            print('Loading QP Faiss Data from Global Area!')
            self.warm_start                 = True
        else:
            print('Loading QP Faiss Data from File!')
            self.warm_start = False
            # gqp.reset()
            self.unset_qp_globals()
            gqp.path       = self.path
            gqp.fname      = self.fname
            gqp.indextype  = self.indextype
            if self.use_s3:
                qp_fname = self.s3_download(ftype='qp')
            else:
                qp_fname = stub + '.qpvars_' + self.indextype + '.npz'
                self.efs_bytes_read += os.path.getsize(qp_fname)                
            with np.load(qp_fname) as data_qp:
                gqp.faiss_index = faiss.deserialize_index(data_qp['FAISS_INDEX'])
                
        # In both cases point instance variables to globals
        self.faiss_index                = gqp.faiss_index
        
        print('DataManager : QP Faiss Data Loaded!')        
        print()
    # ----------------------------------------------------------------------------------------------------------------------------------------    


                             
    
    
