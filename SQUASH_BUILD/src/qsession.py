import numpy as np
from numpy import linalg as LA
import os
import json
import timeit
from datetime import datetime
import faiss
import shutil

class QSession:

    DEBUG = True
    # BITWISE_CONTAINER_DATATYPE = np.uint32         
    # BITWISE_CONTAINER_DATATYPE = np.uint16          # ALSO HELD IN VAQINDEX - KEEP IN SYNC
    BITWISE_CONTAINER_DATATYPE = np.uint8         
    
    def __init__(self, path, fname, mode = "B", num_partitions=None, num_vectors=None, num_dimensions=None, num_attributes=None, num_blocks=1, word_size=4, datatype=np.float32, big_endian=False, \
                 bit_budget=0, attribute_bit_budget=0, non_uniform_bit_alloc=True, design_boundaries=True, bigann=False, vecs_to_print = None):
        
        self.path                       = path
        self.fname                      = fname
        self.mode                       = mode
        self.num_partitions             = num_partitions
        self.num_vectors                = num_vectors
        self.num_dimensions             = num_dimensions
        self.num_attributes             = num_attributes
        self.num_blocks                 = num_blocks
        self.word_size                  = word_size
        self.datatype                   = datatype    
        self.big_endian                 = big_endian
        self.bit_budget                 = bit_budget
        self.attribute_bit_budget       = attribute_bit_budget
        self.non_uniform_bit_alloc      = non_uniform_bit_alloc
        self.design_boundaries          = design_boundaries
        self.bigann                     = bigann
        self.vecs_to_print              = vecs_to_print
        
        self.pipeline                   = []
        self.faiss_buildfuncs           = {}
        self.faiss_infname              = None
        self.faiss_vectors              = None
        self.faiss_indexpath            = None
        self.lloyds_elapsed             = 0

        np.set_printoptions(linewidth=200)
                
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):
       
        # Add derived parameters used in > one class
        self.total_file_words = self.num_vectors * (self.num_dimensions + 1)
        self.num_words_per_block = int(self.total_file_words / self.num_blocks)
        self.num_vectors_per_block = int(self.num_words_per_block / (self.num_dimensions + 1))
        self.tf_num_words_per_block = 0
        self.tf_num_vectors_per_block = 0
        self.tp_num_words_per_block = 0
        
        # Variable arrays used in > one class
        self.dim_means = np.zeros((1, self.num_dimensions), dtype=np.float32)
        self.tf_dim_means = np.zeros((1,self.num_dimensions), dtype=np.float32)
        self.at_means = np.zeros((1,self.num_attributes), dtype=np.float32)
        self.tf_stdevs = np.zeros((1,self.num_dimensions), dtype=np.float32)
        self.at_stdevs = np.zeros((1,self.num_attributes), dtype=np.float32)
        self.cov_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.at_cov_matrix = np.zeros((self.num_attributes, self.num_attributes), dtype=np.float32)
        self.transform_matrix = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.at_transform_matrix = np.zeros((self.num_attributes, self.num_attributes), dtype=np.float32)
        self.cells = None
        self.attribute_cells = None
        self.boundary_vals = None    
        self.attribute_boundary_vals = None
        self.sdc_lookup_lower = None 
        self.sdc_lookup_upper = None
        
        # Attribute Filtering and Binary Quantization
        self.hammings           = None
        self.candidate_count    = None
        self.candidates         = None
        self.candidate_hammings = None

        # Basic validations
        assert os.path.isdir(self.path), "Path entered : " + self.path + " does not exist!"
        assert (self.total_file_words / self.num_blocks) % (self.num_dimensions + 1) == 0, "Inconsistent number of blocks selected."
        assert self.mode in ('B','R','P','X','F'), "Mode must be one of (B)uild, (R)ebuild or (P)rint"      # X is for one-off fix only
        assert self.num_vectors > 0, "num_vectors must be greater than 0"
        assert self.num_dimensions > 0, "num_dimensions must be greater than 0"
        assert self.num_attributes >= 0, "num_attributes cannot be negative"
        assert self.num_blocks > 0, "num_blocks must be greater than 0"
        assert self.word_size > 0, "word_size must be greater than 0"
        assert self.bit_budget > 0, "bit_budget must be greater than 0"
        if self.mode == 'P' and self.vecs_to_print == None:
            print('Mode P requires a list of Vector IDs to be provided!')
            exit(1)
            
        # Print floating-point numbers using a fixed point notation
        np.set_printoptions(suppress=True)
        
        # faiss
        self.faiss_indexpath    = os.path.join(self.path, "faiss")
        self.faiss_infname      = os.path.join(self.path, self.fname)
        self.faiss_buildfuncs   = { "IndexFlatL2": self.build_IndexFlatL2, "IndexSQ8": self.build_IndexSQ8, "IndexPQ": self.build_IndexPQ,  "IF_IVF256_SQ8": self.build_IF_IVF256_SQ8, \
                                    "IndexHNSWFlat": self.build_IndexHNSWFlat, "IF_IVF4096_HNSW32_SQ8": self.build_IF_IVF4096_HNSW32_SQ8, "IndexRaBitQ": self.build_IndexRaBitQ }
        
    #----------------------------------------------------------------------------------------------------------------------------------------        
    def process_timer(self, metric, start_timer):
        # end_timer = timeit.default_timer()
        # duration = end_timer - start_timer
        # update_stats(metric, duration)
        pass
    #----------------------------------------------------------------------------------------------------------------------------------------
    def debug_timer(self, function, reference_time, message, indent=0):
        tabs = ''
        if QSession.DEBUG:
            for i in range(indent + 1):
                tabs += '\t'
            current_time = timeit.default_timer()
            # msg = function + ' -> ' + message
            msg = function.ljust(50,' ') + ' -> ' + message.ljust(50,' ')
            elapsed = tabs + str(current_time - reference_time)
            print("[TIMER] " , msg , "Elapsed: ", elapsed.ljust(20,' '))   
    #----------------------------------------------------------------------------------------------------------------------------------------            
    def isfloat(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False            
    #----------------------------------------------------------------------------------------------------------------------------------------
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
    def build_runner_varsets(self):
        stub = os.path.join(self.path, '') + self.fname
        ptnrvars_present = self.find_file_by_suffix('ptnrvars.npz')
        vaqvars_present  = self.find_file_by_suffix('vaqvars.npz')
        
        if ptnrvars_present:  
            # ---------------------------  
            # Build QueryAllocator varset
            # ---------------------------
            ptnr_fname = stub + '.ptnrvars.npz'
            
            with np.load(ptnr_fname) as data_pt:
                partition_vectors       = data_pt['PART_VECTORS']
                partition_ids           = data_pt['PART_IDS']
                partition_pops          = data_pt['PART_POPS']
                partition_centroids     = data_pt['PART_CENTROIDS']                
            
            as_fname = stub + '.afvars.npz'
            with np.load(as_fname) as data_at:
                at_means                = data_at['AT_MEANS']
                at_stdevs               = data_at['AT_STDEVS']
                attribute_cells         = data_at['AT_CELLS']
                attribute_boundary_vals = data_at['AT_BVALS']                
            
            ds_fname = stub + '.dsvars.npz'
            with np.load(ds_fname) as data_ds:
                dim_means               = data_ds['DIM_MEANS']
                cov_matrix              = data_ds['COV_MATRIX']
                transform_matrix        = data_ds['TRANSFORM_MATRIX']                
        
            # Check if quantized attribute data present. If so, include it
            qattdata_fname = stub + '.afstdq'
            if self.find_file_by_suffix('.afstdq'):
                qattdata = np.fromfile(file=qattdata_fname, count=-1, dtype=np.uint8)
                qattdata = np.reshape(qattdata,(self.num_vectors, self.num_attributes), order="F")
                
                # Write consolidated QA data file - with quantized attribute data
                qavars_fname = stub + '.qavars'
                np.savez(qavars_fname, 
                        PART_VECTORS       = partition_vectors,
                        PART_IDS           = partition_ids,
                        PART_POPS          = partition_pops,
                        PART_CENTROIDS     = partition_centroids,
                        AT_MEANS           = at_means,
                        AT_STDEVS          = at_stdevs,
                        AT_CELLS           = attribute_cells,
                        AT_BVALS           = attribute_boundary_vals,
                        DIM_MEANS          = dim_means,
                        COV_MATRIX         = cov_matrix,
                        TRANSFORM_MATRIX   = transform_matrix,
                        QATT_DATA          = qattdata)                
            else:
                # Write consolidated QA data file - without quantized attribute data
                qavars_fname = stub + '.qavars'
                np.savez(qavars_fname, 
                        PART_VECTORS       = partition_vectors,
                        PART_IDS           = partition_ids,
                        PART_POPS          = partition_pops,
                        PART_CENTROIDS     = partition_centroids,
                        AT_MEANS           = at_means,
                        AT_STDEVS          = at_stdevs,
                        AT_CELLS           = attribute_cells,
                        AT_BVALS           = attribute_boundary_vals,
                        DIM_MEANS          = dim_means,
                        COV_MATRIX         = cov_matrix,
                        TRANSFORM_MATRIX   = transform_matrix)
        
        if vaqvars_present:
            # ---------------------------
            # Build QueryProcessor varset
            # ---------------------------      
            # as_fname = stub + '.afvars.npz'
            # with np.load(as_fname) as data_at:
            #     at_means                = data_at['AT_MEANS']
            #     at_stdevs               = data_at['AT_STDEVS']
            #     attribute_cells         = data_at['AT_CELLS']
            #     attribute_boundary_vals = data_at['AT_BVALS']              
             
            ds_fname = stub + '.dsvars.npz'
            with np.load(ds_fname) as data_ds:
                dim_means               = data_ds['DIM_MEANS']
                cov_matrix              = data_ds['COV_MATRIX']
                transform_matrix        = data_ds['TRANSFORM_MATRIX']   
                
            tf_fname = stub + '.tfvars.npz'
            with np.load(tf_fname) as data_tf:
                tf_dim_means            = data_tf['TF_DIM_MEANS']
                tf_stdevs               = data_tf['TF_STDEVS']                
                
            vaq_fname = stub + '.vaqvars.npz'    
            with np.load(vaq_fname) as data_vq:
                cells                   = data_vq['CELLS']
                boundary_vals           = data_vq['BOUNDARY_VALS']
                sdc_lookup_lower        = data_vq['SDC_LOOKUP_LOWER']
                sdc_lookup_upper        = data_vq['SDC_LOOKUP_UPPER']   
                
            # Also include main VAQ and BQ Indices
            vaqdata_fname = stub + '.vaq'             
            vaqdata = np.fromfile(file=vaqdata_fname, count=-1, dtype=QSession.BITWISE_CONTAINER_DATATYPE)
            container_count = np.uint32(np.ceil(np.divide(self.bit_budget, QSession.BITWISE_CONTAINER_DATATYPE().itemsize * 8)))
            msg = "VAQ file size " + str(vaqdata.shape[0]) + " does not match num_vectors " + str(self.num_vectors) + " times container_count " + str(container_count)
            assert vaqdata.shape[0] == np.dot(self.num_vectors, container_count), msg
            vaqdata = np.reshape(vaqdata,(self.num_vectors, container_count), order="F")

            print("In qsession, QSession.BITWISE_CONTAINER_DATATYPE: ", str(QSession.BITWISE_CONTAINER_DATATYPE))            

            bqdata_fname = stub + '.bq'
            bqdata = np.fromfile(file=bqdata_fname, count=-1, dtype=np.uint8).reshape(self.num_vectors, -1)
            
            # qattdata_fname = stub + '.afstdq'
            # qattdata = np.fromfile(file=qattdata_fname, count=-1, dtype=np.uint8)
            # qattdata = np.reshape(qattdata,(self.num_vectors, self.num_attributes), order="F")            

            # Write consolidated QP data file
            qpvars_fname = stub + '.qpvars'  
            np.savez(qpvars_fname,
                    #  AT_MEANS           = at_means,
                    #  AT_STDEVS          = at_stdevs,
                    #  AT_CELLS           = attribute_cells,
                    #  AT_BVALS           = attribute_boundary_vals,                      
                     DIM_MEANS          = dim_means,
                     COV_MATRIX         = cov_matrix,
                     TRANSFORM_MATRIX   = transform_matrix,
                     TF_DIM_MEANS       = tf_dim_means,
                     TF_STDEVS          = tf_stdevs,
                     CELLS              = cells,
                     BOUNDARY_VALS      = boundary_vals,
                     SDC_LOOKUP_LOWER   = sdc_lookup_lower,
                     SDC_LOOKUP_UPPER   = sdc_lookup_upper,
                     VAQ_DATA           = vaqdata,
                     BQ_DATA            = bqdata )
                    #  QATT_DATA          = qattdata)    
        dummy=0
    # ----------------------------------------------------------------------------------------------------------------------------------------  
    def build_runner_varsets_bigann(self):
        stub = os.path.join(self.path, '') + self.fname
        ptnrvars_present = self.find_file_by_suffix('ptnrvars.npz')
        vaqvars_present  = self.find_file_by_suffix('vaqvars.npz')
        
        if ptnrvars_present:  
            # ---------------------------  
            # Build QueryAllocator varset
            # ---------------------------
            ptnr_fname = stub + '.ptnrvars.npz'
            
            with np.load(ptnr_fname) as data_pt:
                partition_vectors       = data_pt['PART_VECTORS']
                partition_ids           = data_pt['PART_IDS']
                partition_pops          = data_pt['PART_POPS']
                partition_centroids     = data_pt['PART_CENTROIDS']                
            
            labels_fname =  stub + '_sig.npz'
            with np.load(labels_fname) as data_lb:
                lbl_counters        = data_lb['COUNTERS']
                # lbl_vocab_sig       = data_lb['VOCABSIG']
                # lbl_ds_sig          = data_lb['DSSIG']
                lbl_csrt_indices    = data_lb['INDICES']    
                lbl_csrt_indptr     = data_lb['INDPTR']
            
            ds_fname = stub + '.dsvars.npz'
            with np.load(ds_fname) as data_ds:
                dim_means               = data_ds['DIM_MEANS']
                cov_matrix              = data_ds['COV_MATRIX']
                transform_matrix        = data_ds['TRANSFORM_MATRIX']                
        
            # Write consolidated QA data file
            qavars_fname = stub + '.qavars'             # This was missing - not sure why didn't cause an issue?
            np.savez(qavars_fname, 
                    PART_VECTORS       = partition_vectors,
                    PART_IDS           = partition_ids,
                    PART_POPS          = partition_pops,
                    PART_CENTROIDS     = partition_centroids,
                    LBL_COUNTERS       = lbl_counters,
                    # LBL_VOCABSIG       = lbl_vocab_sig,
                    # LBL_DSSIG          = lbl_ds_sig, 
                    LBL_INDICES        = lbl_csrt_indices,
                    LBL_INDPTR         = lbl_csrt_indptr,                    
                    DIM_MEANS          = dim_means,
                    COV_MATRIX         = cov_matrix,
                    TRANSFORM_MATRIX   = transform_matrix)
        
        if vaqvars_present:
            # ---------------------------
            # Build QueryProcessor varset
            # ---------------------------      
            ds_fname = stub + '.dsvars.npz'
            with np.load(ds_fname) as data_ds:
                dim_means               = data_ds['DIM_MEANS']
                cov_matrix              = data_ds['COV_MATRIX']
                transform_matrix        = data_ds['TRANSFORM_MATRIX']   
                
            tf_fname = stub + '.tfvars.npz'
            with np.load(tf_fname) as data_tf:
                tf_dim_means            = data_tf['TF_DIM_MEANS']
                tf_stdevs               = data_tf['TF_STDEVS']                
                
            vaq_fname = stub + '.vaqvars.npz'    
            with np.load(vaq_fname) as data_vq:
                cells                   = data_vq['CELLS']
                boundary_vals           = data_vq['BOUNDARY_VALS']
                sdc_lookup_lower        = data_vq['SDC_LOOKUP_LOWER']
                sdc_lookup_upper        = data_vq['SDC_LOOKUP_UPPER']   
                
            # Also include main VAQ and BQ Indices
            vaqdata_fname = stub + '.vaq'             
            vaqdata = np.fromfile(file=vaqdata_fname, count=-1, dtype=QSession.BITWISE_CONTAINER_DATATYPE)
            container_count = np.uint32(np.ceil(np.divide(self.bit_budget, QSession.BITWISE_CONTAINER_DATATYPE().itemsize * 8)))
            msg = "VAQ file size " + str(vaqdata.shape[0]) + " does not match num_vectors " + str(self.num_vectors) + " times container_count " + str(container_count)
            assert vaqdata.shape[0] == np.dot(self.num_vectors, container_count), msg
            vaqdata = np.reshape(vaqdata,(self.num_vectors, container_count), order="F")

            print("In qsession, QSession.BITWISE_CONTAINER_DATATYPE: ", str(QSession.BITWISE_CONTAINER_DATATYPE))            

            bqdata_fname = stub + '.bq'
            bqdata = np.fromfile(file=bqdata_fname, count=-1, dtype=np.uint8).reshape(self.num_vectors, -1)
            

            # Write consolidated QP data file
            qpvars_fname = stub + '.qpvars'  
            np.savez(qpvars_fname,
                     DIM_MEANS          = dim_means,
                     COV_MATRIX         = cov_matrix,
                     TRANSFORM_MATRIX   = transform_matrix,
                     TF_DIM_MEANS       = tf_dim_means,
                     TF_STDEVS          = tf_stdevs,
                     CELLS              = cells,
                     BOUNDARY_VALS      = boundary_vals,
                     SDC_LOOKUP_LOWER   = sdc_lookup_lower,
                     SDC_LOOKUP_UPPER   = sdc_lookup_upper,
                     VAQ_DATA           = vaqdata,
                     BQ_DATA            = bqdata)

        dummy=0    
    # ----------------------------------------------------------------------------------------------------------------------------------------  
    def build_faiss_runner_varsets(self, model):
        stub = os.path.join(self.path, '') + self.fname
        ptnrvars_present = self.find_file_by_suffix('ptnrvars.npz')
        index_fname      = os.path.join(self.faiss_indexpath, model)
        index_present    = os.path.isfile(index_fname)   
        
        if ptnrvars_present:  
            # ---------------------------  
            # Build QueryAllocator varset
            # ---------------------------
            ptnr_fname = stub + '.ptnrvars.npz'
            
            with np.load(ptnr_fname) as data_pt:
                partition_vectors       = data_pt['PART_VECTORS']
                partition_ids           = data_pt['PART_IDS']
                partition_pops          = data_pt['PART_POPS']
                partition_centroids     = data_pt['PART_CENTROIDS']                
            
            as_fname = stub + '.afvars.npz'
            with np.load(as_fname) as data_at:
                at_means                = data_at['AT_MEANS']
                at_stdevs               = data_at['AT_STDEVS']
                attribute_cells         = data_at['AT_CELLS']
                attribute_boundary_vals = data_at['AT_BVALS']                
        
            # Check if quantized attribute data present. If so, include it
            qattdata_fname = stub + '.afstdq'
            if self.find_file_by_suffix('.afstdq'):
                qattdata = np.fromfile(file=qattdata_fname, count=-1, dtype=np.uint8)
                qattdata = np.reshape(qattdata,(self.num_vectors, self.num_attributes), order="F")
                
                # Write consolidated QA data file - with quantized attribute data
                qavars_fname = stub + '.qavars'
                np.savez(qavars_fname, 
                        PART_VECTORS       = partition_vectors,
                        PART_IDS           = partition_ids,
                        PART_POPS          = partition_pops,
                        PART_CENTROIDS     = partition_centroids,
                        AT_MEANS           = at_means,
                        AT_STDEVS          = at_stdevs,
                        AT_CELLS           = attribute_cells,
                        AT_BVALS           = attribute_boundary_vals,
                        QATT_DATA          = qattdata)                
            else:
                # Write consolidated QA data file - without quantized attribute data
                qavars_fname = stub + '.qavars'
                np.savez(qavars_fname, 
                        PART_VECTORS       = partition_vectors,
                        PART_IDS           = partition_ids,
                        PART_POPS          = partition_pops,
                        PART_CENTROIDS     = partition_centroids,
                        AT_MEANS           = at_means,
                        AT_STDEVS          = at_stdevs,
                        AT_CELLS           = attribute_cells,
                        AT_BVALS           = attribute_boundary_vals )
        
        if index_present:
            # ---------------------------
            # Build QueryProcessor varset
            # ---------------------------      
            # as_fname = stub + '.afvars.npz'
            # with np.load(as_fname) as data_at:
            #     at_means                = data_at['AT_MEANS']
            #     at_stdevs               = data_at['AT_STDEVS']
            #     attribute_cells         = data_at['AT_CELLS']
            #     attribute_boundary_vals = data_at['AT_BVALS']              
                
            # qattdata_fname = stub + '.afstdq'
            # qattdata = np.fromfile(file=qattdata_fname, count=-1, dtype=np.uint8)
            # qattdata = np.reshape(qattdata,(self.num_vectors, self.num_attributes), order="F")            

            # Get faiss index
            faiss_index = faiss.serialize_index(faiss.read_index(index_fname))

            # Write consolidated QP data file
            qpvars_fname = stub + '.qpvars_'   + model
            np.savez(qpvars_fname,
                    #  AT_MEANS           = at_means,
                    #  AT_STDEVS          = at_stdevs,
                    #  AT_CELLS           = attribute_cells,
                    #  AT_BVALS           = attribute_boundary_vals,                      
                     FAISS_INDEX        = faiss_index)    
    # ----------------------------------------------------------------------------------------------------------------------------------------   
    def build_faiss_runner_varsets_bigann(self, model):
        stub = os.path.join(self.path, '') + self.fname
        ptnrvars_present = self.find_file_by_suffix('ptnrvars.npz')
        index_fname      = os.path.join(self.faiss_indexpath, model)
        index_present    = os.path.isfile(index_fname)   
        
        if ptnrvars_present:  
            # ---------------------------  
            # Build QueryAllocator varset
            # ---------------------------
            ptnr_fname = stub + '.ptnrvars.npz'
            
            with np.load(ptnr_fname) as data_pt:
                partition_vectors       = data_pt['PART_VECTORS']
                partition_ids           = data_pt['PART_IDS']
                partition_pops          = data_pt['PART_POPS']
                partition_centroids     = data_pt['PART_CENTROIDS']                
            
            labels_fname =  stub + '_sig.npz'
            with np.load(labels_fname) as data_lb:
                lbl_counters        = data_lb['COUNTERS']
                lbl_csrt_indices    = data_lb['INDICES']    
                lbl_csrt_indptr     = data_lb['INDPTR']
        
            # Write consolidated QA data file
            qavars_fname = stub + '.qavars'
            np.savez(qavars_fname, 
                    PART_VECTORS       = partition_vectors,
                    PART_IDS           = partition_ids,
                    PART_POPS          = partition_pops,
                    PART_CENTROIDS     = partition_centroids,
                    LBL_COUNTERS       = lbl_counters,
                    LBL_INDICES        = lbl_csrt_indices,
                    LBL_INDPTR         = lbl_csrt_indptr )
        
        if index_present:
            # ---------------------------
            # Build QueryProcessor varset
            # ---------------------------   
            # Get faiss index
            faiss_index = faiss.serialize_index(faiss.read_index(index_fname))

            # Write consolidated QP data file
            qpvars_fname = stub + '.qpvars_' + model 
            np.savez(qpvars_fname,
                     FAISS_INDEX        = faiss_index)

        dummy=0        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build_faiss_indexes(self):
        self.recreate_faiss_index_dir()
        self.load_faiss_vectors()
        for model in self.faiss_buildfuncs.keys():
            self.faiss_buildfuncs[model](model)
            if not self.bigann:
                self.build_faiss_runner_varsets(model)
            else:
                self.build_faiss_runner_varsets_bigann(model)            
        dummy=0                
    # ----------------------------------------------------------------------------------------------------------------------------------------      
    def recreate_faiss_index_dir(self):
        # Clear old faiss index folder and contents (if present)
        if os.path.exists(self.faiss_indexpath):
            shutil.rmtree(self.faiss_indexpath)
        # Create new partitions folders
        os.mkdir(self.faiss_indexpath)    
    # ----------------------------------------------------------------------------------------------------------------------------------------          
    def load_faiss_vectors(self):
        self.faiss_vectors     = np.fromfile(file=self.faiss_infname, dtype=self.datatype)
        self.faiss_vectors     = np.reshape(self.faiss_vectors, (self.num_vectors, self.num_dimensions+1), order="C")
        self.faiss_vectors     = np.delete(self.faiss_vectors, 0, 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------      
    # Flat Index
    def build_IndexFlatL2(self, model):
        index_fname = os.path.join(self.faiss_indexpath, model)
        index = faiss.IndexFlatL2(self.num_dimensions)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # SQ8 Index (Filtering Works)
    def build_IndexSQ8(self, model):
        index_fname     = os.path.join(self.faiss_indexpath, model)
        index = faiss.IndexScalarQuantizer(self.num_dimensions,8)
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # RaBitQ Index (Filtering Works)
    def build_IndexRaBitQ(self, model):
        index_fname     = os.path.join(self.faiss_indexpath, model)
        index = faiss.IndexRaBitQ(self.num_dimensions)
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # SQ8 Index (Filtering Works)
    def build_IndexPQ(self, model):
        m               = 16
        nbits           = 8
        index_fname     = os.path.join(self.faiss_indexpath, model)
        index = faiss.IndexPQ(self.num_dimensions, m, nbits)
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # HNSW Flat Index (Filtering - use SearchParametersHNSW)
    def build_IndexHNSWFlat(self, model):
        M               = 32
        ef_construction = 16
        ef_search       = 128
        index_fname     = os.path.join(self.faiss_indexpath, model)

        index = faiss.IndexHNSWFlat(self.num_dimensions, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)    
    # ----------------------------------------------------------------------------------------------------------------------------------------          
    # Index Factory: IVF and SQ8 
    def build_IF_IVF256_SQ8(self, model):
        index_fname = os.path.join(self.faiss_indexpath, model)
        index = faiss.index_factory(self.num_dimensions, "IVF256,SQ8")
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)    
    # ----------------------------------------------------------------------------------------------------------------------------------------              
    # Index Factory: IVF and HNSW (SQ8)
    def build_IF_IVF4096_HNSW32_SQ8(self, model):
        M               = 32
        ef_construction = 16
        ef_search       = 32
        index_fname     = os.path.join(self.faiss_indexpath, model)

        index = faiss.index_factory(self.num_dimensions, "IVF4096_HNSW32,SQ8")
        hnsw = faiss.extract_index_ivf(index)
        hnsw.efConstruction = ef_construction
        hnsw.efSearch = ef_search
        index.train(self.faiss_vectors)
        index.add(self.faiss_vectors)
        faiss.write_index(index, index_fname)    
    # ----------------------------------------------------------------------------------------------------------------------------------------                  
    def run(self):
        
        # Initialisations
        self._initialize()
        
        # Doing these here avoids circular dependency issues
        from attributeset import AttributeSet
        from dataset import DataSet
        from transformed import TransformedDataSet
        from vaqindex import VAQIndex

        print()
        print("Session Begins -> Run Mode ", self.mode)
        print("=============================")
              
        QSession_start_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Session Start Time : ", str(current_time))
        print()

        if self.mode != 'F':

            # Composition classes
            self.AS     = AttributeSet(ctx=self)
            self.DS     = DataSet(ctx=self)
            self.TDS    = TransformedDataSet(ctx=self)
            self.VAQ    = VAQIndex(ctx=self)
        
            # Add Elements to Pipeline. Processing for each Run Mode is controlled within each Element        
            self.pipeline.extend([self.AS, self.DS, self.TDS, self.VAQ])

            # Run the Pipeline
            prev_result = None
            for element in self.pipeline:
                
                class_name = element.__class__.__name__
                start_time = timeit.default_timer()
                print('Pipeline Processing : ', class_name)
                print()
                prev_result = element.process(prev_result)
                self.debug_timer('QSession.run', start_time, class_name + ' processing elapsed time')
                if 'any' in prev_result:
                    if prev_result["any"] == "END":
                        break
                print()                        

            if self.mode in ('B', 'R', 'X'):            # X is for one-off fix only
                if not self.bigann:
                    self.build_runner_varsets()
                else:
                    self.build_runner_varsets_bigann()
            
        else:
            # Faiss index creation mode
            self.build_faiss_indexes()   
            
        QSession_end_time = timeit.default_timer()
        qsession_elapsed = QSession_end_time - QSession_start_time
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
       
        print()
        print("** Lloyds elapsed time for session                    : ", self.lloyds_elapsed, " **")
        print()
        print("Session End Time : ", current_time, " Elapsed : ", str(qsession_elapsed) )
        print()        
        
    #----------------------------------------------------------------------------------------------------------------------------------------


