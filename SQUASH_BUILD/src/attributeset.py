import numpy as np
from numpy import linalg as LA
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed

from base import PipelineElement, TransformationSummary
from qsession import QSession

class AttributeSet(PipelineElement):
    def __init__(self, ctx: QSession = None):
        self.ctx                    = ctx
        self.full_afname            = None      # Columner
        self.full_tp_afname         = None      # Row-wise
        self.full_std_afname        = None      # Columnar
        self.full_quant_std_afname  = None      # Columnar
        self.attribute_energies     = None
        self.aset                   = None
        self.quant_attr_data        = None      # Row-wise

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):
        np.set_printoptions(suppress=True)
        self.full_afname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.af'
        self.full_tp_afname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.aftp'
        self.full_std_afname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.afstd'
        self.full_quant_std_afname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.afstdq'
        
        if self.ctx.mode in ('Q','P'):
            self._load_attribute_vars()
            self._load_quantized_attribute_data()        
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
                f.seek(self.ctx.num_vectors * block_idx * self.ctx.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                if self.ctx.big_endian:
                    block = np.fromfile(file=f, count=self.ctx.num_vectors, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.ctx.num_vectors, dtype=np.float32)

                if block.size > 0:
                    block = np.reshape(block, (self.ctx.num_vectors, 1), order="C")  # Order F to mirror MATLAB  NOT SURE IF NEEDED!
                    yield block
                    block_idx +=1
                else:
                    break     
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate means and stdevs of attribute data (for standardization)
    # Build standardised attributes file. Also build transposed (un-standardised) attributes file as required in P2    
    def _build_attribute_files_old(self):

        gene_ab = self.generate_attribute_block(raw_or_std='raw')
        transposed_attributes = np.zeros((self.ctx.num_vectors, self.ctx.num_attributes), dtype=np.float32)
        
        # Calc means and stdevs. Also build transposed data for P2 reads
        attribute = 0
        with open(self.full_std_afname, mode='wb') as f:
            for ablock in gene_ab:
                self.ctx.at_means[:,attribute] = np.mean(ablock)
                self.ctx.at_stdevs[:,attribute] = np.std(ablock)
                stdat_block =  np.divide(np.subtract(ablock, self.ctx.at_means[:,attribute]), self.ctx.at_stdevs[:,attribute])
                f.write(stdat_block)
                transposed_attributes[:, attribute] = ablock[:,0]
                attribute += 1
            
        # Write out transposed original attribute file
        with open(self.full_tp_afname, mode='wb') as g:
                g.write(transposed_attributes.ravel())                 
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Calculate means and stdevs of attribute data
    # Note that raw attribute file is stored 'columnar' - ie values for attribute 0, values for attribute 1 etc
    # Build transposed (un-standardised) attributes file as required in P2    
    # Build klt-transformed attributes file - also stored 'columnar'.
    def _build_attribute_files(self):

        gene_ab = self.generate_attribute_block(raw_or_std='raw')
        transposed_attributes = np.zeros((self.ctx.num_vectors, self.ctx.num_attributes), dtype=np.float32)
        
        # Calc means and stdevs. Also build transposed data for P2 reads
        with open(self.full_std_afname, mode='wb') as f:
            attribute = 0
            for ablock in gene_ab:
                self.ctx.at_means[:,attribute] = np.mean(ablock)
                self.ctx.at_stdevs[:,attribute] = np.std(ablock)
                transposed_attributes[:, attribute] = ablock[:,0]
                stdat_block =  np.divide(np.subtract(ablock, self.ctx.at_means[:,attribute]), self.ctx.at_stdevs[:,attribute])
                # stdat_block = ablock
                f.write(stdat_block)
                attribute += 1

        # Write out transposed original attribute file
        with open(self.full_tp_afname, mode='wb') as g:
                g.write(transposed_attributes.ravel())    

        # # ALL OF THE BELOW IS DOING KLT ON ATTRIBUTES. HOWEVER, UNABLE TO KLT THE SINGLE-VALUED PREDICATES
        # # DECIDED TO WRITE THE 'STD' FILE (FOR NOW) AS THE STANDARDIZED RAW DATA

        # # Build covariance matrix
        # Y = np.subtract(transposed_attributes,self.ctx.at_means)                
        # self.ctx.at_cov_matrix = np.divide(np.matmul(Y.T, Y), self.ctx.num_vectors)      
            
        # # Calculate eigenvalues (array D) and corresponding eigenvectors (matrix V, one eigenvector per column)
        # D, V = LA.eig(self.ctx.at_cov_matrix)

        # # Sort eigenvalues, while keeping original ordering. 
        # I = np.argsort(D)

        # for i in range(self.ctx.num_attributes):
        #     # Extract eigenvector (looping backwards through original eigenvector ordering).
        #     # Tranpose to make it a row vector
        #     eig_vec = V[:, I[(self.ctx.num_attributes - 1) - i]].T

        #     # Place eigenvector on appropriate row of transform matrix
        #     self.ctx.at_transform_matrix[i, :] = eig_vec

        # # Write transformed attributes file
        # with open(self.full_std_afname, mode="wb") as f:
        #     A = np.subtract(transposed_attributes, self.ctx.at_means)                
        #     B = np.matmul(A, self.ctx.at_transform_matrix).T.ravel()
        #     f.write(B)    # Note we want the transformed attributes to be stored 'columnar' same as the raw attributes
            
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def _save_attribute_vars(self):
        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.afvars', 
                 AT_MEANS               = self.ctx.at_means,
                 AT_STDEVS              = self.ctx.at_stdevs,
                #  AT_COV_MATRIX          = self.ctx.at_cov_matrix,
                #  AT_TRANSFORM_MATRIX    = self.ctx.at_transform_matrix,
                 AT_CELLS               = self.ctx.attribute_cells,
                 AT_BVALS               = self.ctx.attribute_boundary_vals)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_attribute_vars(self):
        with np.load(os.path.join(self.ctx.path, '') + self.ctx.fname + '.afvars.npz') as data:
            self.ctx.at_means                   = data['AT_MEANS']
            self.ctx.at_stdevs                  = data['AT_STDEVS']
            # self.ctx.at_cov_matrix              = data['AT_COV_MATRIX']
            # self.ctx.at_transform_matrix        = data['AT_TRANSFORM_MATRIX']
            self.ctx.attribute_cells            = data['AT_CELLS']
            self.ctx.attribute_boundary_vals    = data['AT_BVALS']
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _transposed_attributes_random_read(self, start_offset, num_words_random_read):
        with open(self.full_tp_afname, mode='rb') as f:
            f.seek(start_offset, os.SEEK_SET)
            block = np.fromfile(file=f, count=num_words_random_read, dtype=np.float32)
            if block.size > 0:
                block = np.reshape(block, (1, self.ctx.num_attributes))
            return block
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _read_vector_attributes(self, vector_id, type='raw'):
        
        if type == 'raw':
            fname = self.full_afname
        elif type == 'std':
            fname = self.full_std_afname
        
        if type in ('raw','std'):
            # offset = np.uint64(vector_id * self.ctx.num_attributes * self.ctx.word_size)        
            # with open(self.full_tp_afname, mode='rb') as f:
            #     f.seek(offset, os.SEEK_SET)
            #     return np.fromfile(file=f, count=self.ctx.num_attributes, dtype=np.float32)
    
            # Requires num_attributes random reads as data is stored 'columnar' (attribute by attribute)
            attribs = np.zeros(self.ctx.num_attributes, dtype=np.float32)
            with open(fname, mode='rb') as f:
                for i in range(self.ctx.num_attributes):
                    offset = np.uint64( ( (i*self.ctx.num_vectors) + vector_id) * self.ctx.word_size )
                    f.seek(offset, os.SEEK_SET)
                    attribs[i] = np.fromfile(file=f, count=1, dtype=np.float32)
            return attribs
    
        elif type == 'qnt':
             return self.quant_attr_data[vector_id,:]
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_attribute_energies(self):
        self.attribute_energies = np.zeros(self.ctx.num_attributes, dtype=np.float32)
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        attribute_count = 0
        for block in ab_gene:       # Each block is (num_vectors, 1)
            self.attribute_energies[attribute_count] = np.var(block)
            attribute_count += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _allocate_attribute_bits(self):
        self.ctx.attribute_cells = np.ones(self.ctx.num_attributes, dtype=np.uint8)  
        if self.ctx.non_uniform_bit_alloc:
            temp_bb = self.ctx.attribute_bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_attribute = np.min(np.argmax(self.attribute_energies))  # np.min to cater for two dims with equal energy - unlikely!

                # Double the number of "cells" for that dimension
                if (self.ctx.attribute_cells[max_energy_attribute] * 2) - 1 > self.ctx.VAQ.MAX_UINT8:
                    pass  # Don't blow the capacity of a UINT8
                else:
                    # self.ctx.attribute_cells[max_energy_attribute] = self.ctx.atribute_cells[max_energy_attribute] * 2 # Orig
                    if self.ctx.attribute_cells[max_energy_attribute] > self.ctx.VAQ.MAX_UINT8 / 2:
                        self.ctx.attribute_cells[max_energy_attribute] = self.ctx.VAQ.MAX_UINT8
                    else:
                        self.ctx.attribute_cells[max_energy_attribute] = self.ctx.attribute_cells[max_energy_attribute] * 2

                # Divide the energy of that dimension by 4 - assumes normal distribution.              
                self.attribute_energies[max_energy_attribute] = self.attribute_energies[max_energy_attribute] / 4

                # Check there aren't more cells than data points (unlikely)
                if self.ctx.attribute_cells[max_energy_attribute] > self.ctx.num_vectors:
                    print("WARNING : self.ctx.attribute_cells[max_energy_attribute] > self.ctx.num_vectors !!")
                    self.ctx.attribute_cells[max_energy_attribute] = self.ctx.attribute_cells[max_energy_attribute] / 2
                else:
                    temp_bb -= 1

        # Uniform bit allocation
        else:
            bits_per_attribute = int(self.ctx.attribute_bit_budget / self.ctx.num_attributes)
            levels = 2 ** bits_per_attribute
            self.ctx.attribute_cells *= levels
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _init_attribute_boundaries(self):
        
        # self.ctx.attribute_boundary_vals = np.zeros((np.max(self.ctx.attribute_cells)+1, self.ctx.num_attributes), dtype=np.float32)
        self.ctx.attribute_boundary_vals = np.zeros((np.uint16(np.max(self.ctx.attribute_cells))+1, self.ctx.num_attributes), dtype=np.float32)
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0

        # Each attribute block is (num_vectors, 1) of np.float32. One block = all values for 1 attribute.
        for block in ab_gene:

            sorted_block = np.sort(block, axis=0)

            # Set first boundary_val (0) along current dimension to just less than min value
            self.ctx.attribute_boundary_vals[0, block_count] = sorted_block[0] - 0.001
            cells_for_attribute = self.ctx.attribute_cells[block_count]

            # Loop over the number of cells allocated to current attribute - careful with indices, should start at 1 and go to penultimate.
            # If cells_for_attribute = 32, this will go to idx 31. That's fine, because attribute_boundary_vals goes up to max(cells) + 1.
            for j in range(1, cells_for_attribute):
                # Using math ceil; alternative is np
                self.ctx.attribute_boundary_vals[j, block_count] = sorted_block[
                    math.ceil(j * self.ctx.num_vectors / cells_for_attribute)]

            # Set final boundary val along current dim
            # Using idx cells_for_dim is safe since attribute_boundary_vals goes up to max(cells) + 1
            self.ctx.attribute_boundary_vals[cells_for_attribute, block_count] = sorted_block[self.ctx.num_vectors - 1] + 0.001

            # Increment block_count
            block_count += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _design_attribute_boundaries(self):
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0

        # If data is PARTITIONED, multiprocessing will occur at the partition level, so don't want to use MP here
        # -------------------------------------------------------------------------------------------------------
        if self.ctx.num_partitions > 0:

            # Loop over blocks (i.e. attributes). Each block is (num_vectors, 1)
            for block in ab_gene:
                # cells_for_attribute = self.ctx.attribute_cells[block_count]
                cells_for_attribute = np.uint16(self.ctx.attribute_cells[block_count])

                # If current attribute only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                # Think values in self.cells are implicitly sorted descending.
                if self.ctx.attribute_cells[block_count] == 1:
                    break

                # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
                d, c = self.ctx.VAQ._lloyd(block_count, block, self.ctx.attribute_boundary_vals[0:cells_for_attribute+1, block_count])
                self.ctx.attribute_boundary_vals[0:cells_for_attribute+1, block_count] = c
                block_count += 1

        else:
            # If data is NOT PARTITIONED, OK to use MP here
            # ---------------------------------------------
            with ProcessPoolExecutor(max_workers=self.ctx.num_attributes) as executor:
                futures = []    
                print("Starting Lloyds (Attribute) Multiprocessing Loop..",flush=True)
                
                # Loop over blocks (i.e. attributes). Each block is (num_vectors, 1)
                for block in ab_gene:
                    # cells_for_attribute = self.ctx.attribute_cells[block_count]
                    cells_for_attribute = np.uint16(self.ctx.attribute_cells[block_count])

                    # If current attribute only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                    # Think values in self.cells are implicitly sorted descending.
                    if self.ctx.attribute_cells[block_count] == 1:
                        break

                    # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                    # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                    # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
                    # r, c = self._lloyd(block_count, block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])
                    futures.append( executor.submit(self.ctx.VAQ._lloyd, dim=block_count, block=block, boundary_vals=self.ctx.attribute_boundary_vals[0:cells_for_attribute+1, block_count] ) )                
                    block_count += 1
            
                returned_attribute = None
                for future in as_completed(futures):
                    for seq, item in enumerate(future.result()):
                        if seq == 0:
                            returned_attribute = item
                            # print('Lloyds complete for Attribute ', item, flush=True)
                        elif seq == 1:
                            # print('Returned Boundary Values : ', item, flush=True)
                            cells_for_attribute = self.ctx.attribute_cells[returned_attribute]
                        self.ctx.attribute_boundary_vals[0:cells_for_attribute+1, returned_attribute] = item
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _create_quantized_attributes_file(self):
        ab_gene = self.generate_attribute_block(raw_or_std='std')
        block_count = 0

        with open(self.full_quant_std_afname, mode='wb') as f:

            # Loop over tp blocks (i.e. loop over dimensions)
            for block in ab_gene:
                self.aset = np.full(self.ctx.num_vectors, -1, dtype=np.int16)   # Changed from int8 to int16 to allow initial value of -1
                for i in range(self.ctx.attribute_cells[block_count]):

                    l = self.ctx.attribute_boundary_vals[i, block_count]
                    r = self.ctx.attribute_boundary_vals[i + 1, block_count]
                    A = np.where(np.logical_and(block >= l, block < r))[0]

                    # MATLAB: Set CSET of those indices to the k-1. Effectively, if a record lies between the 1st and 2nd boundary value, assign it 
                    # to the 0th cells (as this is really the cell bounded by boundary values 1 and 2.)
                    # Python: Set it to k, rather than k-1. If it lies between boundary values 0 and 1, put it in cell 0.
                    self.aset[A] = i

                # Deal with values above max threshold for dimension
                unallocated = np.where(self.aset < 0)[0]
                self.aset[unallocated] = self.ctx.attribute_cells[block_count]

                f.write(np.uint8(self.aset))
                block_count += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Generator to yield quantized attributes
    def _generate_quantized_attributes(self, start_offset=0):
        block_idx = start_offset

        with open(self.full_quant_std_afname, mode="rb") as f:
            while True:
                f.seek(self.ctx.num_vectors * block_idx, os.SEEK_SET)
                block = np.fromfile(file=f, count=self.ctx.num_vectors, dtype=np.uint8)

                if block.size > 0:      
                    yield block.reshape(self.ctx.num_vectors)
                    block_idx += 1
                else:
                    break
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_quantized_attribute_data(self):
        # Load quantized attribute data in ALL cases, even when processing VAQdata from disk
        # if (self.ctx.inmem_vaqdata in ('inmem_oneshot','inmem_columnar')) and (self.ctx.num_attributes > 0):
        if self.ctx.num_attributes > 0:    
            data = np.fromfile(file=self.full_quant_std_afname, count=-1, dtype=np.uint8)
            self.quant_attr_data = np.reshape(data,(self.ctx.num_vectors, self.ctx.num_attributes), order="F")
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build(self):
        self._build_attribute_files()
        self._calc_attribute_energies()
        self._allocate_attribute_bits()
        self._init_attribute_boundaries()
        self._design_attribute_boundaries()
        self._create_quantized_attributes_file()
        self._save_attribute_vars()
        self._load_quantized_attribute_data()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def rebuild(self):
        
        if self.ctx.mode not in ('A','R'):
            print('Attribute Index rebuild called with inappropriate Mode : ',self.ctx.mode)
            exit(1)
        self._load_attribute_vars()
        
        bv_before = np.copy(self.ctx.attribute_boundary_vals)
        self._init_attribute_boundaries()
        if self.ctx.design_boundaries:
            self._design_attribute_boundaries()
        self.ctx.VAQ._compare_boundary_vals(bv_before, self.ctx.attribute_boundary_vals)

        self._save_attribute_vars()
        self._create_quantized_attributes_file()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        
        # if pipe_state != None:
        #     print('PIPELINE ELEMENT AttributeSet : Incoming Pipe State -> ', pipe_state)

        if self.ctx.bigann:
            return {"removed": ("AttributeSet")}

        elif self.ctx.mode in ('F', 'B'):
            self._initialize()
            self.build()
            return {"created": ("AT_MEANS", "AT_STDEVS")}      
        elif self.ctx.mode in ('R'):        
            self._initialize() 
            self.rebuild()
            return {"modified": ("quantized atributes file")}
        elif self.ctx.mode in ('A'):
            self._initialize() 
            self.rebuild()
            return {"modified"  : "quantized attributes file",
                    "any"       : "END"}            
        else:
            self._initialize()            
            return {"instantiated": ("AttributeSet")}                    
    # ----------------------------------------------------------------------------------------------------------------------------------------    
