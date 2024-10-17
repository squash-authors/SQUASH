import numpy as np
from numpy import linalg as LA
from k_means_constrained import KMeansConstrained
import os
import math
import shutil
import timeit
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class Partitioner():
    
    PARTITION_SIZE_VARIATION_PERC = 10       # Want pretty well balanced partitions
    MAX_UINT8 = 255
    MAX_LLOYD_ITERATIONS = 250
    LLOYD_STOP = 0.005                         
    
    def __init__(self, path, fname, mode='B', word_size=4, big_endian=False, attribute_bit_budget=None, non_uniform_bit_alloc=None, design_boundaries=None, \
                    partitioner_blocks=None, num_vectors=None, num_dimensions=None, num_attributes=None, num_partitions=None ):
        
        # Parameter instance variables
        self.path                       = path
        self.fname                      = fname
        self.mode                       = mode
        self.word_size                  = word_size
        self.big_endian                 = big_endian
        self.attribute_bit_budget       = attribute_bit_budget
        self.non_uniform_bit_alloc      = non_uniform_bit_alloc
        self.design_boundaries          = design_boundaries        
        self.partitioner_blocks         = partitioner_blocks
        self.num_vectors                = num_vectors
        self.num_dimensions             = num_dimensions
        self.num_attributes             = num_attributes
        self.num_partitions             = num_partitions

        # Non-parameter instance variables
        self.full_fname                 = None
        self.full_tf_fname              = None
        self.full_afname                = None
        self.partition_vectors          = None
        self.labels                     = None     
        self.tf_data                    = None   
        self.ds_data                    = None   
        self.at_data                    = None
        self.partitions_root            = None
        self.clf                        = None
        self.partition_ids              = None
        self.partition_pops             = None
        self.partition_centroids        = None
        self.attribute_cells            = None
        self.attribute_energies         = None
        
        # To enable build_tf
        self.at_means                   = None
        self.at_stdevs                  = None
        self.dim_means                  = None
        self.cov_matrix                 = None
        self.transform_matrix           = None    
        self.tf_dim_means               = None
        self.tf_stdevs                  = None
        self.total_file_words           = None
        self.num_words_per_block        = None
        self.num_vectors_per_block      = None
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):
        np.set_printoptions(suppress=True)
        self.full_fname                 = os.path.join(self.path, '') + self.fname
        self.full_tf_fname              = os.path.join(self.path, '') + self.fname + '.tf'
        self.full_afname                = os.path.join(self.path, '') + self.fname + '.af'
        self.full_tp_afname             = os.path.join(self.path, '') + self.fname + '.aftp'
        self.full_std_afname            = os.path.join(self.path, '') + self.fname + '.afstd'
        self.full_quant_std_afname      = os.path.join(self.path, '') + self.fname + '.afstdq'        
        self.partitions_root            = os.path.join(self.path, '') + 'partitions'
        # self.partition_vectors          = np.zeros((self.num_vectors, self.num_partitions), dtype=np.int16)
        self.partition_vectors          = np.zeros((self.num_vectors, self.num_partitions), dtype=np.uint8)
        
        self.at_means               = np.zeros((1,self.num_attributes),dtype=np.float32)
        self.at_stdevs              = np.zeros((1,self.num_attributes),dtype=np.float32)
        self.dim_means              = np.zeros((self.num_dimensions),dtype=np.float32)
        self.cov_matrix             = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)
        self.transform_matrix       = np.zeros((self.num_dimensions, self.num_dimensions), dtype=np.float32)   
        self.tf_dim_means           = np.zeros((1,self.num_dimensions),dtype=np.float32)
        self.tf_stdevs              = np.zeros((1,self.num_dimensions),dtype=np.float32)
             
        self.total_file_words       = self.num_vectors * (self.num_dimensions + 1)
        self.num_words_per_block    = int(self.total_file_words / self.partitioner_blocks)
        self.num_vectors_per_block  = int(self.num_words_per_block / (self.num_dimensions + 1))
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _preprocess_attributeset(self):
        # This is required because QueryAllocator requires AttributeSet vars file and quantized attribute data
        self._build_attribute_files()
        self._calc_attribute_energies()
        self._allocate_attribute_bits()
        self._init_attribute_boundaries()
        self._design_attribute_boundaries()
        self._create_quantized_attributes_file()
    #----------------------------------------------------------------------------------------------------------------------------------------
    # Read all values for one attribute
    def generate_attribute_block(self, start_offset=0, raw_or_std='raw'):
        block_idx = start_offset      
        if raw_or_std == 'raw':
            fna = self.full_afname
        elif raw_or_std == 'std':
            fna = self.full_std_afname
        else:
            print('generate_attribute_block -> Invalid value supplied for raw_or_std!')
            exit(1)
        
        with open(fna, mode="rb") as f:
            
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
    # Calculate means and stdevs of attribute data
    # Note that raw attribute file is stored 'columnar' - ie values for attribute 0, values for attribute 1 etc
    # Build transposed (un-standardised) attributes file as required in P2    
    # Build klt-transformed attributes file - also stored 'columnar'.
    def _build_attribute_files(self):
        gene_ab = self.generate_attribute_block(raw_or_std='raw')
        transposed_attributes = np.zeros((self.num_vectors, self.num_attributes), dtype=np.float32)
        
        # Calc means and stdevs. Also build transposed data for P2 reads
        with open(self.full_std_afname, mode='wb') as f:
            attribute = 0
            for ablock in gene_ab:
                self.at_means[:,attribute] = np.mean(ablock)
                self.at_stdevs[:,attribute] = np.std(ablock)
                transposed_attributes[:, attribute] = ablock[:,0]
                stdat_block =  np.divide(np.subtract(ablock, self.at_means[:,attribute]), self.at_stdevs[:,attribute])
                # stdat_block = ablock
                f.write(stdat_block)
                attribute += 1

        # Write out transposed original attribute file
        with open(self.full_tp_afname, mode='wb') as g:
                g.write(transposed_attributes.ravel())    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_attribute_energies(self):
        self.attribute_energies = np.zeros(self.num_attributes, dtype=np.float32)
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        attribute_count = 0
        for block in ab_gene:       # Each block is (num_vectors, 1)
            self.attribute_energies[attribute_count] = np.var(block)
            attribute_count += 1
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _allocate_attribute_bits(self):
        self.attribute_cells = np.ones(self.num_attributes, dtype=np.uint8)  
        if self.non_uniform_bit_alloc:
            temp_bb = self.attribute_bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_attribute = np.min(np.argmax(self.attribute_energies))  # np.min to cater for two dims with equal energy - unlikely!

                # Double the number of "cells" for that dimension
                if (self.attribute_cells[max_energy_attribute] * 2) - 1 > Partitioner.MAX_UINT8:
                    pass  # Don't blow the capacity of a UINT8
                else:
                    if self.attribute_cells[max_energy_attribute] > Partitioner.MAX_UINT8 / 2:
                        self.attribute_cells[max_energy_attribute] = Partitioner.MAX_UINT8
                    else:
                        self.attribute_cells[max_energy_attribute] = self.attribute_cells[max_energy_attribute] * 2

                # Divide the energy of that dimension by 4 - assumes normal distribution.              
                self.attribute_energies[max_energy_attribute] = self.attribute_energies[max_energy_attribute] / 4

                # Check there aren't more cells than data points (unlikely)
                if self.attribute_cells[max_energy_attribute] > self.num_vectors:
                    print("WARNING : self.attribute_cells[max_energy_attribute] > self.num_vectors !!")
                    self.attribute_cells[max_energy_attribute] = self.attribute_cells[max_energy_attribute] / 2
                else:
                    temp_bb -= 1

        # Uniform bit allocation
        else:
            bits_per_attribute = int(self.attribute_bit_budget / self.num_attributes)
            levels = 2 ** bits_per_attribute
            self.attribute_cells *= levels
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _init_attribute_boundaries(self):
        self.attribute_boundary_vals = np.zeros((np.max(self.attribute_cells)+1, self.num_attributes), dtype=np.float32)
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0

        # Each attribute block is (num_vectors, 1) of np.float32. One block = all values for 1 attribute.
        for block in ab_gene:

            sorted_block = np.sort(block, axis=0)

            # Set first boundary_val (0) along current dimension to just less than min value
            self.attribute_boundary_vals[0, block_count] = sorted_block[0] - 0.001
            cells_for_attribute = self.attribute_cells[block_count]

            # Loop over the number of cells allocated to current attribute - careful with indices, should start at 1 and go to penultimate.
            # If cells_for_attribute = 32, this will go to idx 31. That's fine, because attribute_boundary_vals goes up to max(cells) + 1.
            for j in range(1, cells_for_attribute):
                # Using math ceil; alternative is np
                self.attribute_boundary_vals[j, block_count] = sorted_block[
                    math.ceil(j * self.num_vectors / cells_for_attribute)]

            # Set final boundary val along current dim
            # Using idx cells_for_dim is safe since attribute_boundary_vals goes up to max(cells) + 1
            self.attribute_boundary_vals[cells_for_attribute, block_count] = sorted_block[self.num_vectors - 1] + 0.001

            # Increment block_count
            block_count += 1
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _design_attribute_boundaries(self):
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []    
            print("Starting Multiprocessing Loop..",flush=True)
            
            # Loop over blocks (i.e. attributes). Each block is (num_vectors, 1)
            for block in ab_gene:
                cells_for_attribute = self.attribute_cells[block_count]

                # If current attribute only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                # Values in self.cells are implicitly sorted descending.
                if self.attribute_cells[block_count] == 1:
                    break

                # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
                # r, c = self._lloyd(block_count, block, self.boundary_vals[0:cells_for_dim+1, block_count])
                futures.append( executor.submit(self._lloyd, dim=block_count, block=block, boundary_vals=self.attribute_boundary_vals[0:cells_for_attribute+1, block_count] ) )                
                block_count += 1
        
        returned_attribute = None
        for future in as_completed(futures):
            for seq, item in enumerate(future.result()):
                if seq == 0:
                    returned_attribute = item
                    # print('Lloyds complete for Attribute ', item, flush=True)
                elif seq == 1:
                    # print('Returned Boundary Values : ', item, flush=True)
                    cells_for_attribute = self.attribute_cells[returned_attribute]
                    self.attribute_boundary_vals[0:cells_for_attribute+1, returned_attribute] = item
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _lloyd(self, dim, block, boundary_vals):
        # Inputs: 
        # TSET -> block (num_vectors, 1)
        # B(1:CELLS(i)+1,i) -> self.boundary_vals[0:cells_for_dim+1, block_count].
        # First param is the whole unsorted block. Also gets all initialized boundary values for current dim.
        # Dimensions of boundary_vals_in: (cells(i)+1, 1)

        delta = np.inf
        c = boundary_vals
        M1 = np.min(boundary_vals)
        M2 = np.max(boundary_vals)
        num_boundary_vals = np.shape(boundary_vals)[0]
        r = np.zeros(num_boundary_vals, dtype=np.float32)

        num_lloyd_iterations = 0
        while True:
            delta_new = np.float32(0)
            num_lloyd_iterations += 1

            # Loop over intervals; careful with indices
            for i in range(num_boundary_vals - 1):         
                # Find values in block between boundary values
                X_i = block[np.where(np.logical_and(block >= c[i], block < c[i + 1]))]

                if np.shape(X_i)[0] > 0:
                    r[i] = np.mean(X_i)
                else:
                    r[i] = np.random.rand(1) * (M2 - M1) + M1

                # Add representation error over current interval to delta_new
                delta_new += np.sum(np.square(X_i - r[i]))

            # Sort representative values - todo: sorting algorithm selection
            r = np.sort(r)

            # Update boundary values based on representative values
            # for j in range(1, num_boundary_vals):  # MATLAB has a -1 here... don't think we need?   YES WE DO! Otherwise lose top boundary
            for j in range(1, num_boundary_vals - 1):

                c[j] = (r[j - 1] + r[j]) / 2

            # Stopping condition check
            if ( np.abs( ((delta - delta_new) / delta) ) < Partitioner.LLOYD_STOP ) or ( num_lloyd_iterations >= Partitioner.MAX_LLOYD_ITERATIONS ):
                print("Number of Lloyd's iterations: ", str(num_lloyd_iterations), flush=True)
                return dim, c
            delta = delta_new    
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _create_quantized_attributes_file(self):
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0
        with open(self.full_quant_std_afname, mode='wb') as f:
            # Loop over tp blocks (i.e. loop over dimensions)
            for block in ab_gene:
                self.aset = np.full(self.num_vectors, -1, dtype=np.int16)   # Changed from int8 to int16 to allow initial value of -1
                for i in range(self.attribute_cells[block_count]):

                    l = self.attribute_boundary_vals[i, block_count]
                    r = self.attribute_boundary_vals[i + 1, block_count]
                    A = np.where(np.logical_and(block >= l, block < r))[0]

                    # MATLAB: Set CSET of those indices to the k-1. Effectively, if a record lies between the 1st and 2nd boundary value, assign it 
                    # to the 0th cells (as this is really the cell bounded by boundary values 1 and 2.)
                    # Python: Set it to k, rather than k-1. If it lies between boundary values 0 and 1, put it in cell 0.
                    self.aset[A] = i

                # Deal with values above max threshold for dimension
                unallocated = np.where(self.aset < 0)[0]
                self.aset[unallocated] = self.attribute_cells[block_count]

                f.write(np.uint8(self.aset))
                block_count += 1
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _save_attribute_vars(self):
        np.savez(os.path.join(self.path, '') + self.fname + '.afvars', 
                 AT_MEANS               = self.at_means,
                 AT_STDEVS              = self.at_stdevs,
                 AT_CELLS               = self.attribute_cells,
                 AT_BVALS               = self.attribute_boundary_vals)
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _load_tf_data(self):
        # Populate in-memory Vectors array
        total_tf_words = (self.num_vectors * self.num_dimensions)
        with open(self.full_tf_fname, mode="rb") as f:
            self.tf_data = np.fromfile(file=f, count=total_tf_words, dtype=np.float32)
            if self.tf_data.size > 0:
                self.tf_data = np.reshape(self.tf_data, (self.num_vectors, self.num_dimensions), order="C")
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _unload_tf_data(self):
        self.tf_data = None
        del self.tf_data
    #----------------------------------------------------------------------------------------------------------------------------------------    
    def _load_ds_data(self):
        # Populate in-memory dataset for partitioning
        total_dataset_words = (self.num_vectors * (self.num_dimensions + 1))
        with open(self.full_fname, mode="rb") as f:
            if self.big_endian:
                self.ds_data = np.fromfile(file=f, count=total_dataset_words, dtype=np.float32).byteswap(inplace=True)
            else:
                self.ds_data = np.fromfile(file=f, count=total_dataset_words, dtype=np.float32)
            if self.ds_data.size > 0:
                self.ds_data = np.reshape(self.ds_data, (self.num_vectors, self.num_dimensions+1), order="C")
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def _load_at_data(self):
        # Populate in-memory Attributes array
        total_attributeset_words = (self.num_vectors * self.num_attributes)
        with open(self.full_afname, mode="rb") as g:
            
            if self.big_endian:
                self.at_data = np.fromfile(file=g, count=total_attributeset_words, dtype=np.float32).byteswap(inplace=True)
            else:
                self.at_data = np.fromfile(file=g, count=total_attributeset_words, dtype=np.float32)

            if self.at_data.size > 0:
                # self.at_data = np.reshape(self.at_data, (self.num_vectors, self.num_attributes), order="C")
                self.at_data = np.reshape(self.at_data, (self.num_vectors, self.num_attributes), order="F")
            dummy=0
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _recreate_partition_dirs(self):
        # Clear old partitions folders and data (if present)
        if os.path.exists(self.partitions_root):
            shutil.rmtree(self.partitions_root)

        # Create new partitions folders
        os.mkdir(self.partitions_root)
        for partition_no in range (self.num_partitions):
            part_dir = os.path.join(self.partitions_root, str(partition_no))
            os.mkdir(part_dir)
    # ----------------------------------------------------------------------------------------------------------------------------------------            
    def _run_kmc(self):
        kmc_start_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("KMeansConstrained Start Time : ", str(current_time))
        
        partition_size = (self.num_vectors // self.num_partitions)
        partition_min_size = np.uint32(partition_size - (partition_size * (Partitioner.PARTITION_SIZE_VARIATION_PERC / 100)))
        partition_max_size = np.uint32(partition_size + (partition_size * (Partitioner.PARTITION_SIZE_VARIATION_PERC / 100)))
        
        self.clf = KMeansConstrained(
                                        n_clusters      =   self.num_partitions,
                                        size_min        =   partition_min_size,
                                        size_max        =   partition_max_size,
                                        random_state    =   0,                          # If int, seed used by the random number generator. If None, RandomState instance used by np.random
                                        init            =   'k-means++',                # selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
                                        # max_iter        =   300,                        # Maximum number of iterations of the k-means algorithm for a single run.
                                        max_iter        =   30,                        # Maximum number of iterations of the k-means algorithm for a single run.
                                        tol             =   0.0001,                     # Relative tolerance w.r.t Frobenius norm of the diff in the cluster centers of two consec iters. to declare convergence.
                                        verbose         =   1,                          # Verbosity mode
                                        copy_x          =   True,                       # Ensures additional data is not changed at all (involves taking a safe copy)
                                        n_jobs          =   -1                          # -1 = Use all cpus, 1 = No parallel computing, n_jobs = -2, all CPUs but one are used, etc 
                                    )        
        
        self.clf.fit_predict(self.tf_data)
        # self.clf.fit(self.tf_data)
        
        self.labels                 = self.clf.labels_
        self.partition_centroids    = self.clf.cluster_centers_
        self.partition_ids, self.partition_pops = np.unique(self.labels, return_counts=True)

        for p_id, p_pop in zip(self.partition_ids, self.partition_pops):
            print(f"Partition {p_id} has {p_pop} Vectors")        
            
        kmc_end_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("KMeansConstrained End Time : ", current_time, " Elapsed : ", str(kmc_end_time - kmc_start_time) )
        print()                
        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _create_subsets(self):
        for p_id in self.partition_ids:
            
            partpath                = os.path.join(os.path.join(self.path,'partitions'),str(p_id))
            part_dataset_fname      = os.path.join(partpath, self.fname)
            part_attributeset_fname = os.path.join(partpath, self.fname) + '.af'
            
            with open(part_dataset_fname, 'wb') as fd, open(part_attributeset_fname, 'wb') as fa:
                
                inds = np.where(self.labels == p_id)[0]
                self.ds_data[inds].tofile(fd)
                # self.at_data[inds].tofile(fa)
                self.at_data[inds].ravel(order='F').tofile(fa)                
                self.partition_vectors[inds,p_id] = 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_partitioner_vars(self):
        # np.savez(os.path.join(self.path, '') + self.fname + '.ptnrvars', PART_VECTORS=self.partition_vectors, 
        #                 PART_IDS=self.partition_ids, PART_POPS=self.partition_pops, PART_CENTROIDS=self.partition_centroids)

        # NB Saving partition_vectors in columnar packed format to reduce memory reqs at runtime   
        if self.partition_vectors.shape[0] == self.num_vectors:
            # Not already packed - pack
            pvecs = np.packbits(self.partition_vectors, axis=0)
        else:
            pvecs = self.partition_vectors
            
        np.savez(os.path.join(self.path, '') + self.fname + '.ptnrvars', PART_VECTORS=pvecs, 
                        PART_IDS=self.partition_ids, PART_POPS=self.partition_pops, PART_CENTROIDS=self.partition_centroids)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_partitioner_vars(self):
        print("Loading partitioner variables from ", self.path)
        with np.load(os.path.join(self.path, '') + self.fname + '.ptnrvars.npz') as data:
            self.partition_vectors  = data['PART_VECTORS']
            self.partition_ids      = data['PART_IDS']
            self.partition_pops     = data['PART_POPS']
            self.partition_centroids = data['PART_CENTROIDS']
    # ----------------------------------------------------------------------------------------------------------------------------------------                
    def _generate_dataset_block(self,start_offset=0):
        block_idx = start_offset
        with open(self.full_fname, mode="rb") as f:
            while True:
                f.seek(self.num_words_per_block*block_idx*self.word_size, os.SEEK_SET) 
                if self.big_endian:
                    block = np.fromfile(file=f, count=self.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.num_words_per_block, dtype=np.float32)

                if block.size > 0:
                    block = np.reshape(block, (self.num_vectors_per_block, self.num_dimensions+1), order="C")
                    block = np.delete(block, 0, 1)
                    yield block
                    block_idx +=1
                else:
                    break
    # ----------------------------------------------------------------------------------------------------------------------------------------     
    def _preprocess_dataset(self):
        # Create dim_means for full dataset
        gene1 = self._generate_dataset_block()       
        for block1 in gene1:        
            dim_sums = np.divide(np.sum(block1, axis=0).reshape(1, self.num_dimensions), self.num_vectors_per_block)
            self.dim_means = np.add(self.dim_means, dim_sums)
        self.dim_means = np.divide(self.dim_means, self.partitioner_blocks).astype(np.float32)        
    
        # Create cov_matrix for full dataset        
        gene2 = self._generate_dataset_block()
        for block2 in gene2:
            Y = np.subtract(block2, self.dim_means)                
            self.cov_matrix = self.cov_matrix + np.divide(np.matmul(Y.T, Y), self.num_vectors_per_block)
        self.cov_matrix = np.divide(self.cov_matrix, self.partitioner_blocks).astype(np.float32)        
    
        # Create transform_matrix for full dataset
        D, V = LA.eig(self.cov_matrix)
        I = np.argsort(D)
        for i in range(self.num_dimensions):
            eig_vec = V[:, I[(self.num_dimensions - 1) - i]].T
            self.transform_matrix[i, :] = eig_vec
    # ----------------------------------------------------------------------------------------------------------------------------------------     
    def _save_dataset_vars(self):
        np.savez(os.path.join(self.path, '') + self.fname + '.dsvars', DIM_MEANS=self.dim_means,
                 COV_MATRIX=self.cov_matrix, TRANSFORM_MATRIX=self.transform_matrix)    
    # ----------------------------------------------------------------------------------------------------------------------------------------         
    def _build_tf(self):
        with open(self.full_tf_fname, mode="wb") as f:
            gene = self._generate_dataset_block()       
            for X in gene:
                Y = np.subtract(X, self.dim_means)                
                Z = np.matmul(Y, self.transform_matrix)
                f.write(Z)
    # ----------------------------------------------------------------------------------------------------------------------------------------     
    def _calc_tf_dim_means_stdevs(self):
        # NB Needs _load_tf_data method to have run first!
        # Calc dim_means and stdevs
        for dim in range(self.num_dimensions):
            self.tf_dim_means[:,dim] = np.mean(self.tf_data[:,dim])
            self.tf_stdevs[:,dim]    = np.std(self.tf_data[:,dim])
            
        # # Write out standardized transformed file
        # gene_tf = self.generate_tf_block()
        
        # with open(self.full_tfbq_fname, mode='wb') as f:
        #     for tf_block in gene_tf:
        #         tfbq_block =  np.divide(np.subtract(tf_block, self.tf_dim_means), self.tf_stdevs)
        #         f.write(tfbq_block)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_tf_vars(self):
        np.savez(os.path.join(self.path, '') + self.fname + '.tfvars', TF_DIM_MEANS=self.tf_dim_means,
                 TF_STDEVS=self.tf_stdevs)                        
    # ----------------------------------------------------------------------------------------------------------------------------------------  
    def _find_file_by_suffix(self, suffix):
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
    def _build_qa_vars(self):
        stub = os.path.join(self.path, '') + self.fname
        ptnrvars_present = self._find_file_by_suffix('ptnrvars.npz')
        
        if ptnrvars_present:    
            # Build QueryAllocator varset
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
            qadata_fname = stub + '.afstdq'
            if self._find_file_by_suffix('.afstdq'):
                qadata = np.fromfile(file=qadata_fname, count=-1, dtype=np.uint8)
                qadata = np.reshape(qadata,(self.num_vectors, self.num_attributes), order="F")
                
                # Write consoiildated QA data file - with quantized attribute data
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
                        QATT_DATA          = qadata)                
            else:
                # Write consoiildated QA data file - without quantized attribute data
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
        dummy = 0
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    def process(self):
        
        partitioner_start_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Partitioner Start Time : ", str(current_time))
        print()
        
        if (self.mode == 'B') and (self.num_partitions > 0):
            self._initialize()
            self._recreate_partition_dirs()
            self._preprocess_attributeset()
            self._save_attribute_vars()            
            self._preprocess_dataset()
            self._save_dataset_vars()  
            self._build_tf()            
            self._load_tf_data()    
            self._calc_tf_dim_means_stdevs()
            self._save_tf_vars()             
            self._run_kmc()
            self._unload_tf_data()
            self._load_ds_data()
            self._load_at_data()
            self._create_subsets()
            self._save_partitioner_vars()
            self._build_qa_vars()
        elif (self.mode == 'R') and (self.num_partitions > 0):
            self._initialize()
            self._build_qa_vars()            
            self._load_partitioner_vars()
        elif (self.mode == 'X') and (self.num_partitions > 0):      # One-off, to enable rebuild of ptrnr_vars and qa_vars with packed partition_vectors array
            self._initialize()
            self._load_partitioner_vars() 
            self._save_partitioner_vars()                         
            self._build_qa_vars()            
            self._load_partitioner_vars()            
        elif self.num_partitions > 0:
            self._initialize()
            self._load_partitioner_vars()
        else:
            pass

        partitioner_end_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Partitioner End Time : ", current_time, " Elapsed : ", str(partitioner_end_time - partitioner_start_time) )
        print()    

        if self.num_partitions > 0:
            return self.partition_pops
        

            
    # ----------------------------------------------------------------------------------------------------------------------------------------