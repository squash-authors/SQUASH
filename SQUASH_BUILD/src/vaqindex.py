import numpy as np
import os
import math
import timeit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed

from base import PipelineElement, TransformationSummary
from qsession import QSession

class VAQIndex(PipelineElement):
    
    MAX_UINT8 = 255
    MAX_LLOYD_ITERATIONS = 250
    LLOYD_STOP = 0.005
    WEIGHTING_FACTOR = 0.1  
    MAX_PROCESSES = 16
    # BITWISE_CONTAINER_DATATYPE = np.uint32            
    # BITWISE_CONTAINER_DATATYPE = np.uint16              # NB ALSO IN QSESSION - KEEP IN SYNC!
    BITWISE_CONTAINER_DATATYPE = np.uint8

    def __init__(self, ctx: QSession = None):
        self.ctx                  = ctx
        self.full_vaq_fname       = None 
        self.bq_fname             = None  
        self.energies             = None
        self.cset                 = None
        self.vaqdata              = None
        self.bqdata               = None  

        self._initialize()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):

        self.full_vaq_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.vaq'
        self.bq_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.bq'        
        if self.ctx.non_uniform_bit_alloc == False:
            assert self.ctx.bit_budget % self.ctx.num_dimensions == 0, "Bit budget cannot be evenly divided among dimensions (uniform bit allocation)."
            
        # For query-only runs, load CELLS and BOUNDARY_VALS from file saved during VAQ build
        if self.ctx.mode in ('Q','P'):
            self._load_vaq_vars()
            self._load_vaqdata()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transformed file 
    def _calc_energies_old(self):

        self.energies = np.zeros(self.ctx.num_dimensions, dtype=np.float32)
        tf_gene = self.ctx.TDS.generate_tf_block()

        for block in tf_gene:       # Each block is (num_dims, num_vectors_per_block)
            block = np.square(block)

            # Sum along columns -> add to energies. Energies is (1,num_dims)
            self.energies += np.sum(block, axis=0)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transformed file 
    def _calc_energies(self):

        self.energies = np.zeros(self.ctx.num_dimensions, dtype=np.float32)
        tp_gene = self.ctx.TDS.generate_tp_block()
        block_idx = 0

        for block in tp_gene:       # Each block is (num_dims, num_vectors_per_block)
            block_var = np.var(block)
            self.energies[block_idx] = block_var
            block_idx += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def _allocate_bits(self):

        self.ctx.cells = np.ones(self.ctx.num_dimensions, dtype=np.uint8)  
        if self.ctx.non_uniform_bit_alloc:
            temp_bb = self.ctx.bit_budget
            while temp_bb > 0:
                # Get index of dimension with maximum energy
                max_energy_dim = np.min(np.argmax(self.energies))  # np.min to cater for two dims with equal energy - unlikely!

                # Double the number of "cells" for that dimension
                if (self.ctx.cells[max_energy_dim] * 2) - 1 > VAQIndex.MAX_UINT8: # Extra cells
                # if self.ctx.cells[max_energy_dim] * 2 > VAQIndex.MAX_UINT8: # Orig
                    pass  # Don't blow the capacity of a UINT8
                else:
                    # self.ctx.cells[max_energy_dim] = self.ctx.cells[max_energy_dim] * 2 # Orig
                    if self.ctx.cells[max_energy_dim] > VAQIndex.MAX_UINT8 / 2:
                        self.ctx.cells[max_energy_dim] = VAQIndex.MAX_UINT8
                    else:
                        self.ctx.cells[max_energy_dim] = self.ctx.cells[max_energy_dim] * 2

                # Divide the energy of that dimension by 4 - assumes normal distribution.              
                self.energies[max_energy_dim] = self.energies[max_energy_dim] / 4

                # Check there aren't more cells than data points (unlikely)
                if self.ctx.cells[max_energy_dim] > self.ctx.num_vectors:
                    print("WARNING : self.ctx.cells[max_energy_dim] > self.ctx.num_vectors !!")
                    self.ctx.cells[max_energy_dim] = self.ctx.cells[max_energy_dim] / 2
                else:
                    temp_bb -= 1

        # Uniform bit allocation
        else:
            bits_per_dim = int(self.ctx.bit_budget / self.ctx.num_dimensions)
            levels = 2 ** bits_per_dim
            self.ctx.cells *= levels
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Uses transposed file. Initializes boundary values such that cells are equally populated.
    def _init_boundaries(self):

        self.ctx.boundary_vals = np.zeros((np.max(self.ctx.cells)+1, self.ctx.num_dimensions), dtype=np.float32)
        tp_gene = self.ctx.TDS.generate_tp_block()
        block_count = 0

        # Each tp block is (num_vectors, 1) of np.float32. One block = all values for 1 dimension.
        for block in tp_gene:

            sorted_block = np.sort(block, axis=0)

            # Set first boundary_val (0) along current dimension to just less than min value
            self.ctx.boundary_vals[0, block_count] = sorted_block[0] - 0.001
            cells_for_dim = self.ctx.cells[block_count]

            # Loop over the number of cells allocated to current dimension - careful with indices, should start at 1 and go to penultimate.
            # If cells_for_dim = 32, this will go to idx 31. That's fine, because boundary_vals goes up to max(cells) + 1.
            for j in range(1, cells_for_dim):
                # Using math ceil; alternative is np
                self.ctx.boundary_vals[j, block_count] = sorted_block[
                    math.ceil(j * self.ctx.num_vectors / cells_for_dim)]

            # Set final boundary val along current dim
            # Using idx cells_for_dim is safe since boundary_vals goes up to max(cells) + 1
            self.ctx.boundary_vals[cells_for_dim, block_count] = sorted_block[self.ctx.num_vectors - 1] + 0.001

            # Increment block_count
            block_count += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on transposed file
    def _design_boundaries(self):
    
        tp_gene = self.ctx.TDS.generate_tp_block()
        block_count = 0

        # If data is PARTITIONED, multiprocessing will occur at the partition level, so don't want to use MP here
        # -------------------------------------------------------------------------------------------------------
        if self.ctx.num_partitions > 0:
            # Loop over blocks (i.e. dimensions). Each block is (num_vectors, 1)
            for block in tp_gene:
                cells_for_dim = self.ctx.cells[block_count]

                # If current dimension only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                # Think values in self.cells are implicitly sorted descending.
                if self.ctx.cells[block_count] == 1:
                    break

                # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
                d, c = self._lloyd(block_count, block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])
                self.ctx.boundary_vals[0:cells_for_dim+1, block_count] = c
                block_count += 1        
        
        else:
            # If data is NOT PARTITIONED, OK to use MP here
            # ---------------------------------------------
            with ProcessPoolExecutor(max_workers=VAQIndex.MAX_PROCESSES) as executor:
            # with ProcessPoolExecutor() as executor:
                futures = []    
                print("Starting Lloyds (VAQ) Multiprocessing Loop..",flush=True)
                
                # Loop over blocks (i.e. dimensions). Each block is (num_vectors, 1)
                for block in tp_gene:
                    cells_for_dim = self.ctx.cells[block_count]

                    # If current dimension only has 1 cell (i.e. 0 bits allocated to it), then break and end.
                    # Think values in self.cells are implicitly sorted descending.
                    if self.ctx.cells[block_count] == 1:
                        break

                    # Call Lloyd's algorithm function -> could be replaced by modified Lloyds
                    # MATLAB has B(1:CELLS(i)+1, i). Say a dim has 4 cells, this goes from 1 to 5, inclusive.
                    # Ours will go from 0 to 5, not inclusive at the top, so really 0,1,2,3,4. Therefore equivalent. 
                    # r, c = self._lloyd(block_count, block, self.ctx.boundary_vals[0:cells_for_dim+1, block_count])
                    futures.append( executor.submit(self._lloyd, dim=block_count, block=block, boundary_vals=self.ctx.boundary_vals[0:cells_for_dim+1, block_count] ) )                
                    block_count += 1
            
                returned_dim = None
                for future in as_completed(futures):
                    for seq, item in enumerate(future.result()):
                        if seq == 0:
                            returned_dim = item
                            # print('Lloyds complete for Dimension ', item, flush=True)
                        elif seq == 1:
                            # print('Returned Boundary Values : ', item, flush=True)
                            cells_for_dim = self.ctx.cells[returned_dim]
                            self.ctx.boundary_vals[0:cells_for_dim+1, returned_dim] = item
        
        # Generate SDC Lookup
        print('Generating SDC Lookup Table')
        max_cell_count = np.uint64(np.max(self.ctx.cells))
        max_boundary_count = max_cell_count + np.uint64(1)
        self.ctx.sdc_lookup_lower = np.zeros((np.dot(max_cell_count, max_cell_count), self.ctx.num_dimensions), dtype=np.float32)
        self.ctx.sdc_lookup_upper = np.zeros((np.dot(max_cell_count, max_cell_count), self.ctx.num_dimensions), dtype=np.float32)
        
        for i in range(self.ctx.num_dimensions):
            cells_for_current_dim = self.ctx.cells[i]
            for j in range(max_cell_count): # where query lands 
                for k in range(cells_for_current_dim): # finding distance to all other cells
                    if j < k:
                        self.ctx.sdc_lookup_lower[np.uint32((j*max_cell_count) + k), i] = np.square(np.subtract(self.ctx.boundary_vals[k,i], self.ctx.boundary_vals[(j+1),i]))
                        self.ctx.sdc_lookup_upper[np.uint32((j*max_cell_count) + k), i] = np.square(np.subtract(self.ctx.boundary_vals[(k+1),i], self.ctx.boundary_vals[j,i]))
                    elif j == k:
                        self.ctx.sdc_lookup_lower[np.uint32((j*max_cell_count) + k), i] = 0
                        self.ctx.sdc_lookup_upper[np.uint32((j*max_cell_count) + k), i] = np.square(np.subtract(self.ctx.boundary_vals[(k+1),i], self.ctx.boundary_vals[j,i]))
                    else:
                        self.ctx.sdc_lookup_lower[np.uint32((j*max_cell_count) + k), i] = np.square(np.subtract(self.ctx.boundary_vals[j,i], self.ctx.boundary_vals[(k+1),i]))
                        self.ctx.sdc_lookup_upper[np.uint32((j*max_cell_count) + k), i] = np.square(np.subtract(self.ctx.boundary_vals[(j+1),i], self.ctx.boundary_vals[k,i]))
    # ----------------------------------------------------------------------------------------------------------------------------------------
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
            if ( np.abs( ((delta - delta_new) / delta) ) < VAQIndex.LLOYD_STOP ) or ( num_lloyd_iterations >= VAQIndex.MAX_LLOYD_ITERATIONS ):
                print("Number of Lloyd's iterations: ", str(num_lloyd_iterations), flush=True)
                return dim, c
            delta = delta_new
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on the transposed datafile. 
    def _create_vaqfile_old(self):

        tp_gene = self.ctx.TDS.generate_tp_block()
        block_count = 0

        with open(self.full_vaq_fname, mode='wb') as f:

            # Loop over tp blocks (i.e. loop over dimensions)
            for block in tp_gene:
                self.cset = np.full(self.ctx.num_vectors, -1, dtype=np.int16)   # Changed from int8 to int16
                for i in range(self.ctx.cells[block_count]):

                    l = self.ctx.boundary_vals[i, block_count]
                    r = self.ctx.boundary_vals[i + 1, block_count]
                    A = np.where(np.logical_and(block >= l, block < r))[0]

                    # MATLAB: Set CSET of those indices to the k-1. Effectively, if a record lies between the 1st and 2nd boundary value, assign it 
                    # to the 0th cells (as this is really the cell bounded by boundary values 1 and 2.)
                    # Python: Set it to k, rather than k-1. If it lies between boundary values 0 and 1, put it in cell 0.
                    self.cset[A] = i

                # Deal with values above max threshold for dimension
                unallocated = np.where(self.cset < 0)[0]
                self.cset[unallocated] = self.ctx.cells[block_count]

                f.write(np.uint8(self.cset))
                block_count += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Operates on the transposed datafile. 
    def _create_vaqfile(self):

        dt              = VAQIndex.BITWISE_CONTAINER_DATATYPE
        dt_bits         = dt().itemsize * 8      
        num_vecs        = np.uint32(self.ctx.num_vectors)
        segment         = np.zeros(num_vecs,dtype=dt)
        segment_residue = np.zeros(num_vecs,dtype=dt)
        seg_used        = np.uint8(0)
        bit_allocs      = np.uint8(np.ceil(np.log2(self.ctx.cells)))
        residual_bits   = np.uint8(0)
        byte_count      = np.uint32(0)

        print("In _create_vaqfile(), dt: ", str(dt))

        tp_gene = self.ctx.TDS.generate_tp_block()
        block_count = 0

        with open(self.full_vaq_fname, mode='wb') as f:

            # Loop over tp blocks (i.e. loop over dimensions)
            for block in tp_gene:
                self.cset = np.zeros(num_vecs, dtype=dt)
                for i in range(self.ctx.cells[block_count]):

                    l = self.ctx.boundary_vals[i, block_count]
                    r = self.ctx.boundary_vals[i + 1, block_count]
                    A = np.where(np.logical_and(block >= l, block < r))[0]
                    self.cset[A] = i

                while True:
                    if seg_used == dt_bits:
                        print(segment)    
                        f.write(segment)
                        byte_count += (num_vecs * 4)
                        segment  = np.zeros(num_vecs,dtype=dt)
                        seg_used = 0
                        # break
                        continue
                        
                    elif (seg_used == 0) and (residual_bits == 0):
                        segment = np.bitwise_or(segment, np.left_shift(self.cset,(dt_bits - bit_allocs[block_count]),dtype=dt))
                        print(np.binary_repr(segment[0],dt_bits))
                        seg_used += bit_allocs[block_count]
                        break
                    
                    elif (seg_used == 0) and (residual_bits > 0):
                        segment  = np.bitwise_or(segment, segment_residue)
                        print(np.binary_repr(segment[0],dt_bits))
                        seg_used += residual_bits
                        segment_residue = np.zeros(num_vecs,dtype=dt)
                        residual_bits = 0

                    elif (seg_used > 0) and ((dt_bits - seg_used) >= bit_allocs[block_count]):
                        segment = np.bitwise_or(segment, np.left_shift(self.cset,(dt_bits - seg_used - bit_allocs[block_count]),dtype=dt))
                        print(np.binary_repr(segment[0],dt_bits))
                        seg_used += bit_allocs[block_count]
                        break
                                
                    elif (seg_used > 0) and ((dt_bits - seg_used) < bit_allocs[block_count]):
                        residual_bits = bit_allocs[block_count] - (dt_bits - seg_used)
                        segment_residue = np.left_shift(self.cset, (dt_bits - residual_bits), dtype=dt)
                        segment  = np.bitwise_or(segment, np.right_shift(self.cset,(bit_allocs[block_count] - (dt_bits - seg_used)),dtype=dt))
                        print(np.binary_repr(segment[0],dt_bits))
                        seg_used     += (bit_allocs[block_count] - residual_bits)
                        break   # New                
                
                block_count +=1    

            # Deal with end case
            if np.sum(segment) > 0:
                print(segment)
                f.write(segment)
            # elif np.sum(segment_residue) > 0:
            if np.sum(segment_residue) > 0:            
                print(segment_residue)
                f.write(segment_residue)
            # else:
            #     pass    # Everything already written to file

    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def _create_bitfile(self):
        
        bq_bytes_per_vector = np.uint8(np.ceil(np.divide(self.ctx.num_dimensions, 8)))
        self.bqdata = np.zeros([self.ctx.num_vectors, bq_bytes_per_vector], dtype=np.uint8)

        tfbq_gene = self.ctx.TDS.generate_tfbq_block()
        offset = 0
        vecs_per_block = self.ctx.tf_num_vectors_per_block
        
        # Read tf file in blocks
        for block in tfbq_gene:
            
            # Vectorized np.where to create array of 1/0 uint8s, then feed into np.packbits
            self.bqdata[(offset * vecs_per_block):((offset * vecs_per_block) + vecs_per_block),:] = np.packbits(np.where(block<=0, 0, 1),axis=1)
            offset += 1

        # Create output file
        with open(self.bq_fname, mode='wb') as f:
            f.write(self.bqdata)         
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)     
    def generate_vaq_block_old(self, start_offset=0):

        block_idx = start_offset

        # Reading a column of VAQ index per block. Each word (usually 4 bytes) contains 4 VAQ values
        with open(self.full_vaq_fname, mode="rb") as f:
            while True:
                f.seek(self.ctx.tp_num_words_per_block * block_idx, os.SEEK_SET)
                block = np.fromfile(file=f, count=self.ctx.tp_num_words_per_block, dtype=np.uint8)

                if block.size > 0:      
                    # if self.ctx.binary_quantization:
                    if self.ctx.candidate_count > 0:
                        yield block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                    else:
                        yield block.reshape(self.ctx.num_vectors)
                    block_idx += 1
                else:
                    break
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)     
    def generate_vaq_block(self):

        dt              = VAQIndex.BITWISE_CONTAINER_DATATYPE
        dt_bits         = dt().itemsize * 8 
        num_vecs        = self.ctx.num_vectors
        bit_allocs      = np.uint8(np.ceil(np.log2(self.ctx.cells)))
        segment         = np.zeros(num_vecs,dtype=dt)
        segment_residue = np.zeros(num_vecs,dtype=dt)
        seg_used        = np.uint8(0)
        residual_bits   = np.uint8(0)
        dim             = 0
        block_idx       = 0

        with open(self.full_vaq_fname, mode="rb") as f:

            while dim < self.ctx.num_dimensions:
                # f.seek(self.ctx.tp_num_words_per_block * block_idx, os.SEEK_SET)
                f.seek((self.ctx.tp_num_words_per_block * dt().itemsize * block_idx), os.SEEK_SET)
                block = np.fromfile(file=f, count=self.ctx.tp_num_words_per_block, dtype=dt)
                if block.size == 0:
                    break
                
                while dim < self.ctx.num_dimensions:
                    
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
                            if self.ctx.candidate_count > 0:
                                yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                            else:
                                yield out_block.reshape(self.ctx.num_vectors)
                    
                    elif seg_used == 0:
                        out_block = np.right_shift(block,dt_bits - bit_allocs[dim])
                        # print(np.binary_repr(out_block[0],dt_bits))
                        # print(out_block)                
                        seg_used += bit_allocs[dim]
                        dim += 1
                        if self.ctx.candidate_count > 0:
                            yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                        else:
                            yield out_block.reshape(self.ctx.num_vectors)
                        
                    elif (seg_used > 0) and (bit_allocs[dim] <= (dt_bits - seg_used)):
                        out_block = np.right_shift(np.left_shift(block,seg_used),dt_bits - bit_allocs[dim])
                        # print(np.binary_repr(out_block[0],dt_bits))
                        # print(out_block)                
                        seg_used += bit_allocs[dim]
                        dim += 1
                        if self.ctx.candidate_count > 0:
                            yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                        else:
                            yield out_block.reshape(self.ctx.num_vectors)

                    elif (seg_used > 0) and (bit_allocs[dim] > (dt_bits - seg_used)):                
                        segment_residue = np.right_shift(np.left_shift(block,seg_used), dt_bits - bit_allocs[dim])
                        residual_bits = dt_bits - seg_used
                        seg_used += residual_bits
                        block_idx += 1
                        break            # Next block     
    
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)       
    def generate_vaq_block_mem_old(self, start_offset=0):

        block_idx = start_offset

        # Reading a column (dimension) of VAQ index from in-memory self self.vaqdata. 
        while block_idx < self.ctx.num_dimensions:
            # if self.ctx.binary_quantization:
            if self.ctx.candidate_count > 0:
                yield self.vaqdata[self.ctx.candidates,block_idx].reshape(self.ctx.candidate_count)
            else:
                yield self.vaqdata[:,block_idx].reshape(self.ctx.num_vectors)
            block_idx += 1
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of vaq_index; all data for a single dimension (or all data for a candidate set)
    def generate_vaq_block_mem(self):

        dt              = VAQIndex.BITWISE_CONTAINER_DATATYPE
        dt_bits         = dt().itemsize * 8 
        num_vecs        = self.ctx.num_vectors
        bit_allocs      = np.uint8(np.ceil(np.log2(self.ctx.cells)))
        segment         = np.zeros(num_vecs,dtype=dt)
        segment_residue = np.zeros(num_vecs,dtype=dt)
        seg_used        = np.uint8(0)
        residual_bits   = np.uint8(0)
        dim             = 0
        block_idx       = 0

        while dim < self.ctx.num_dimensions:
            block = self.vaqdata[:,block_idx].reshape(self.ctx.num_vectors)
            while dim < self.ctx.num_dimensions:
                
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
                        if self.ctx.candidate_count > 0:
                            yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                        else:
                            yield out_block.reshape(self.ctx.num_vectors)
                
                elif seg_used == 0:
                    out_block = np.right_shift(block,dt_bits - bit_allocs[dim])
                    # print(np.binary_repr(out_block[0],dt_bits))
                    # print(out_block)                
                    seg_used += bit_allocs[dim]
                    dim += 1
                    if self.ctx.candidate_count > 0:
                        yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                    else:
                        yield out_block.reshape(self.ctx.num_vectors)
                    
                elif (seg_used > 0) and (bit_allocs[dim] <= (dt_bits - seg_used)):
                    out_block = np.right_shift(np.left_shift(block,seg_used),dt_bits - bit_allocs[dim])
                    # print(np.binary_repr(out_block[0],dt_bits))
                    # print(out_block)                
                    seg_used += bit_allocs[dim]
                    dim += 1
                    if self.ctx.candidate_count > 0:
                        yield out_block[self.ctx.candidates].reshape(self.ctx.candidate_count)
                    else:
                        yield out_block.reshape(self.ctx.num_vectors)

                elif (seg_used > 0) and (bit_allocs[dim] > (dt_bits - seg_used)):                
                    segment_residue = np.right_shift(np.left_shift(block,seg_used), dt_bits - bit_allocs[dim])
                    residual_bits = dt_bits - seg_used
                    seg_used += residual_bits
                    block_idx += 1
                    break            # Next block
    
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # Read one vector from VAQIndex.  
    def _read_vaq_vector_old(self, vec_id=None):

        # If inmem_vaqdata, simply read from array.    
        if self.ctx.inmem_vaqdata in ('inmem_oneshot', 'inmem_columnar'):
            return self.vaqdata[vec_id,:]
        
        # Otherwise read from disk (needs a loop as stored by dimension, not vector)
        qvec = np.zeros(self.ctx.num_dimensions, dtype=np.uint8)
        with open(self.full_vaq_fname, mode="rb") as f:
            for dim_no in range(self.ctx.num_dimensions):
                offset = (dim_no * self.ctx.num_vectors) + vec_id
                f.seek(offset, os.SEEK_SET)
                qvec[dim_no] = np.fromfile(file=f, count=1, dtype=np.uint8)
        return qvec   
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Read one vector from VAQIndex (Enhanced for bitwise processing)
    def _read_vaq_vector(self, vec_id=None):

        dim_bitallocs   = np.uint8(np.log2(self.ctx.cells))
        elements        = []    
        bitsequence     = ""
        dt              = VAQIndex.BITWISE_CONTAINER_DATATYPE
        dt_wordsize     = dt().itemsize
        dt_size         = dt_wordsize * 8


        # If inmem_vaqdata, read from array and extract dimension values from container variables
        if self.ctx.inmem_vaqdata in ('inmem_oneshot', 'inmem_columnar'):
            for val in self.vaqdata[vec_id,:]:
                bitsequence += np.binary_repr(val,dt_size)

            startpos = np.uint64(0)
            for bits in dim_bitallocs:
                elements.append(np.uint8(int(bitsequence[startpos:startpos+bits],2))) 
                startpos += bits
            return np.asarray(elements, dtype=np.uint8)
        
        # Otherwise read from disk (needs a loop as stored by dimension, not vector. Also need to extract dims from container variables)
        num_vecs = self.ctx.num_vectors
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
    def _save_vaq_vars(self):
        
        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.vaqvars', 
                 CELLS=self.ctx.cells, BOUNDARY_VALS=self.ctx.boundary_vals, SDC_LOOKUP_LOWER=self.ctx.sdc_lookup_lower, SDC_LOOKUP_UPPER=self.ctx.sdc_lookup_upper)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaq_vars(self):

        vaq_full_varfile = os.path.join(self.ctx.path, '') + self._find_file_by_suffix('.vaqvars.npz')
        print("Loading vaq variables from ", vaq_full_varfile)
        with np.load(vaq_full_varfile) as data:
            self.ctx.cells = data['CELLS']
            self.ctx.boundary_vals = data['BOUNDARY_VALS']
            self.ctx.sdc_lookup_lower = data['SDC_LOOKUP_LOWER']
            self.ctx.sdc_lookup_upper = data['SDC_LOOKUP_UPPER']
    #----------------------------------------------------------------------------------------------------------------------------------------
    def _find_file_by_suffix(self, suffix):

        hit_count = 0
        hits = []
        for file in os.listdir(self.ctx.path):
            if file.endswith(suffix):
                hits.append(file)
                hit_count += 1
        if hit_count > 1:
            raise ValueError("Too many hits for suffix ", str(suffix))
        else:
            return hits[0]
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaqdata_old(self):

        if self.ctx.inmem_vaqdata in ('inmem_oneshot','inmem_columnar'):
            data = np.fromfile(file=self.full_vaq_fname, count=-1, dtype=np.uint8)
            self.vaqdata = np.reshape(data,(self.ctx.num_vectors, self.ctx.num_dimensions), order="F")
            print()
            if self.ctx.inmem_vaqdata == 'inmem_oneshot':
                msg = '(ONESHOT) IN-MEMORY VAQ PROCESSING SELECTED!'
            elif self.ctx.inmem_vaqdata == 'inmem_columnar':
                msg = '(COLUMNAR) IN-MEMORY VAQ PROCESSING SELECTED!'
            print(msg)
            print()
        
        if self.ctx.binary_quantization:
            self.bqdata = np.fromfile(file=self.bq_fname, count=-1, dtype=np.uint8).reshape(self.ctx.num_vectors, -1)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_vaqdata(self):

        if self.ctx.inmem_vaqdata in ('inmem_oneshot','inmem_columnar'):
            data = np.fromfile(file=self.full_vaq_fname, count=-1, dtype=VAQIndex.BITWISE_CONTAINER_DATATYPE)
            container_count = np.uint32(np.ceil(np.divide(self.ctx.bit_budget, VAQIndex.BITWISE_CONTAINER_DATATYPE().itemsize * 8)))
            msg = "VAQ file size " + str(data.shape[0]) + " does not match num_vectors " + str(self.ctx.num_vectors) + " times container_count " + str(container_count)
            assert data.shape[0] == np.dot(self.ctx.num_vectors, container_count), msg
            self.vaqdata = np.reshape(data,(self.ctx.num_vectors, container_count), order="F")
            print()
            if self.ctx.inmem_vaqdata == 'inmem_oneshot':
                msg = '(ONESHOT) IN-MEMORY VAQ PROCESSING SELECTED!'
            elif self.ctx.inmem_vaqdata == 'inmem_columnar':
                msg = '(COLUMNAR) IN-MEMORY VAQ PROCESSING SELECTED!'
            print(msg)
            print()
        
        if self.ctx.binary_quantization:
            self.bqdata = np.fromfile(file=self.bq_fname, count=-1, dtype=np.uint8).reshape(self.ctx.num_vectors, -1)
            
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def build(self):
        
        self._calc_energies()
        self._allocate_bits()

        self._init_boundaries()
        if self.ctx.design_boundaries:
            self._design_boundaries()

        self._save_vaq_vars()
        self._create_vaqfile()

        if self.ctx.mode == 'F':
            self._load_vaqdata()
            
        # Create binary quantization file
        self._create_bitfile()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Mode R: Full rebuild of VAQ file
    def rebuild(self):
                
        if self.ctx.mode != 'R':
            print('VAQIndex rebuild called with inappropriate Mode : ',self.ctx.mode)
            exit(1)
        
        # Load CELLS data (ignore BOUNDARY_VALS - will be re-initialised)
        self._load_vaq_vars()

        # Take copy of current boundary vals for comparison
        bv_before = np.copy(self.ctx.boundary_vals)
        
        # Need to do this again - bit budget may have changed
        self._calc_energies()
        self._allocate_bits()        

        # Init boundary values
        self._init_boundaries()

        # Design boundary values (with Lloyd's)
        if self.ctx.design_boundaries:
            self._design_boundaries()

        # See if boundary vals have changed
        # self._compare_boundary_vals(bv_before, self.ctx.boundary_vals)

        # Save cells and boundary_vals for use elsewhere
        self._save_vaq_vars()

        # Create vaqfile
        self._create_vaqfile()
        
        # Create binary quantization file
        self._create_bitfile()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Compare two sets of boundary values
    def _compare_boundary_vals(self, bv1, bv2):
        
        if not self.ctx.DEBUG:
            return
        
        if bv1.shape != bv2.shape:
            print('First boundary array is ', bv1.shape, ' Second boundary array is ', bv2.shape, ' MISMATCH!')
            exit(1)
        
        rtol = 1e-05
        atol = 1e-08
        if np.allclose(bv1, bv2, rtol, atol, equal_nan=True):
            print()
            print("Boundary Value sets match!")
            print()
            return
        
        # At least one mismatch. Loop through dims and identify changed values
        np.set_printoptions(suppress=True)
        for dim in range(bv1.shape[1]):
            if not np.allclose(bv1[:,dim], bv2[:,dim], rtol, atol, True):
                wanted = np.invert(np.isclose(bv1[:,dim], bv2[:,dim], rtol, atol, True))
                bv1_unmatched = bv1[wanted, dim]
                bv2_unmatched = bv2[wanted, dim]
                print('Dimension : ', dim, '  -> Boundary Vals mismatched')
                print('Before:')
                print(bv1_unmatched.T)
                print('After:')
                print(bv2_unmatched.T)
                print()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Compare two floats for almost-equalness
    def _fleq(self, fl1, fl2):

        rtol = 1e-05
        atol = 1e-08
        return np.isclose(fl1, fl2, rtol, atol, True)

    #----------------------------------------------------------------------------------------------------------------------------------------
    # Print details of one or more vectors
    def prvd(self):

        # Loop over requested vectors
        for vec_id in self.ctx.vecs_to_print:
            
            print("Details for Vector ID : ", vec_id)
            print()
            print('ATTRIBUTES')

            # Read vector attributes (raw and standardized) from AttributeSet
            atraw = self.ctx.AS._read_vector_attributes(vec_id,'raw')
            atstd = self.ctx.AS._read_vector_attributes(vec_id,'std')
            atqnt = self.ctx.AS._read_vector_attributes(vec_id,'qnt')
            
            print('{:^4s}     {:^10s} {:^10s} {:^4s}    {:<10s}   {:<10s}  {:<6s}  {:>20s} '.format('Attr', 'raw', 'std', 'qnt', 'Boundary', 'LB', 'UB', 'Surrounding Boundaries'))
            # Loop over Attributes            
            for att_no in range(self.ctx.num_attributes):
                boundary_start = (atqnt[att_no] - 4) if (atqnt[att_no] - 4) > 0 else 0
                boundary_stop  = (atqnt[att_no] + 5) if (atqnt[att_no] + 5) <= self.ctx.attribute_cells[att_no] else self.ctx.attribute_cells[att_no] + 1
                B1 = np.abs(np.subtract(atstd[att_no], self.ctx.attribute_boundary_vals[atqnt[att_no], att_no]))
                B2 = np.abs(np.subtract(atstd[att_no], self.ctx.attribute_boundary_vals[atqnt[att_no]+1, att_no]))
                LB = min(B1,B2)
                UB = max(B1,B2)

                print('{:^4d}  {:>10.4f} {:>10.4f}   {:>4d} {:>10.4f} {:>10.4f} {:>10.4f}    '.format(att_no, atraw[att_no], atstd[att_no], atqnt[att_no], self.ctx.attribute_boundary_vals[atqnt[att_no], att_no], LB, UB), end='')
                print(self.ctx.attribute_boundary_vals[boundary_start:boundary_stop, att_no])            
            
                                
            # Read vector from TransformedDataSet
            start_offset = vec_id * self.ctx.num_dimensions * self.ctx.word_size
            tvec = self.ctx.TDS.tf_random_read(start_offset, self.ctx.num_dimensions) 
                        
            # Read quantised vector from VAQIndex
            qvec = self._read_vaq_vector(vec_id)
            
            print()
            print('DIMENSIONS')
            print('{:^4s}     {:^10s} {:^4s}          {:<10s}        {:<10s}   {:<6s} {:>20s} '.format('Dim', 'tvec', 'qvec', 'Boundary', 'LB', 'UB', 'Surrounding Boundaries'))
            # Loop over Dimensions            
            for dim_no in range(self.ctx.num_dimensions):
                boundary_start = (qvec[dim_no] - 4) if (qvec[dim_no] - 4) > 0 else 0
                boundary_stop  = (qvec[dim_no] + 5) if (qvec[dim_no] + 5) <= self.ctx.num_dimensions else self.ctx.num_dimensions + 1
                B1 = np.abs(np.subtract(tvec[0,dim_no], self.ctx.boundary_vals[qvec[dim_no], dim_no]))
                B2 = np.abs(np.subtract(tvec[0,dim_no], self.ctx.boundary_vals[qvec[dim_no]+1, dim_no]))
                LB = min(B1,B2)
                UB = max(B1,B2)
                # print('{:^4d}   {:>10.4f} {:>4d}    {:>10.4f}      '.format(dim_no, tvec[0,dim_no], qvec[dim_no], self.ctx.boundary_vals[qvec[dim_no], dim_no]), end='')
                
                print('{:^4d}   {:>10.4f} {:>4d}         {:>10.4f}     {:>10.4f} {:>10.4f}    '.format(dim_no, tvec[0,dim_no], qvec[dim_no], self.ctx.boundary_vals[qvec[dim_no], dim_no], LB, UB), end='')
                print(self.ctx.boundary_vals[boundary_start:boundary_stop, dim_no])
            
            print()
    #----------------------------------------------------------------------------------------------------------------------------------------
    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:

        # if pipe_state != None:
        #     print('PIPELINE ELEMENT VAQIndex : Incoming Pipe State -> ', pipe_state)
            
        if self.ctx.mode in ('F', 'B'):
            self.build()
            return {"created": ("VAQ_FILE")}     
        elif self.ctx.mode in ('R'):
            self.rebuild()
            return {"modified": ("VAQ_FILE")}             
        elif self.ctx.mode in ('P'):
            self.prvd()
            return {"extra": ("Vector Details Print")}                         
        else:
            return {"instantiated": ("VAQIndex")}                
    #----------------------------------------------------------------------------------------------------------------------------------------
    
