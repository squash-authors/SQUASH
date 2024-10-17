import numpy as np
from numpy import linalg as LA
import os

from base import PipelineElement, TransformationSummary
from qsession import QSession

class DataSet(PipelineElement):
    def __init__(self, ctx: QSession = None):
        self.ctx            = ctx
        self.full_fname     = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):
        np.set_printoptions(suppress=True)
        self.full_fname = os.path.join(self.ctx.path, '') + self.ctx.fname

    #----------------------------------------------------------------------------------------------------------------------------------------
    def _read_block(self, block_idx, f_handle):

        #(num_dims+1, block_size/(num_dims+1)) -> (num_dims, num_vectors_per_block) if all as float32

        f_handle.seek(self.ctx.num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
        if self.ctx.big_endian:
            block = np.fromfile(file=f_handle, count=self.ctx.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
        else:
            block = np.fromfile(file=f_handle, count=self.ctx.num_words_per_block, dtype=np.float32)

        block = np.reshape(block, (self.ctx.num_vectors_per_block, self.ctx.num_dimensions+1), order="C")
        block = np.delete(block, 0, 1)
        
        return block
    #----------------------------------------------------------------------------------------------------------------------------------------
    def generate_dataset_block(self,start_offset=0):

        block_idx = start_offset
        with open(self.full_fname, mode="rb") as f:
            
            while True:
                f.seek(self.ctx.num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET) # Multiply by word_size since seek wants a byte location.
                if self.ctx.big_endian:
                    block = np.fromfile(file=f, count=self.ctx.num_words_per_block, dtype=np.float32).byteswap(inplace=True)
                else:
                    block = np.fromfile(file=f, count=self.ctx.num_words_per_block, dtype=np.float32)

                if block.size > 0:
                    block = np.reshape(block, (self.ctx.num_vectors_per_block, self.ctx.num_dimensions+1), order="C")
                    block = np.delete(block, 0, 1)
                    # AT THIS POINT, EACH COL IS A VECTOR AND EACH ROW IS A DIMENSION (FOR 1/NUM_BLOCKS OF THE VECTORS)

                    yield block
                    block_idx +=1
                else:
                    break
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_dim_means(self):

        with open(self.full_fname, mode='rb') as f:
            for i in range(self.ctx.num_blocks):
                block = self._read_block(i, f)
                dim_sums = np.divide(np.sum(block, axis=0).reshape(1, self.ctx.num_dimensions), self.ctx.num_vectors_per_block)
                self.ctx.dim_means = np.add(self.ctx.dim_means, dim_sums)

        self.ctx.dim_means = np.divide(self.ctx.dim_means, self.ctx.num_blocks).astype(np.float32) # Oh yes it did
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # self.cov_matrix is (num_dimensions, num_dimensions)
    def _calc_cov_matrix(self):

        with open(self.full_fname, mode='rb') as f:
            for i in range(self.ctx.num_blocks):
                # Read block -> gives (num_dimensions, num_vectors_per_block) matrix
                X = self._read_block(i, f)
                Y = np.subtract(X, self.ctx.dim_means)                
                self.ctx.cov_matrix = self.ctx.cov_matrix + np.divide(np.matmul(Y.T, Y), self.ctx.num_vectors_per_block)

        self.ctx.cov_matrix = np.divide(self.ctx.cov_matrix, self.ctx.num_blocks).astype(np.float32)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Transformation matrix is (num_dimensions, num_dimensions)
    def _calc_transform_matrix(self):

        # Calculate eigenvalues (array D) and corresponding eigenvectors (matrix V, one eigenvector per column)
        # D is already equivalent to E in MATLAB code.
        D, V = LA.eig(self.ctx.cov_matrix)

        # Sort eigenvalues, while keeping original ordering. 
        I = np.argsort(D)

        for i in range(self.ctx.num_dimensions):
            # Extract eigenvector (looping backwards through original eigenvector ordering).
            # Tranpose to make it a row vector
            eig_vec = V[:, I[(self.ctx.num_dimensions - 1) - i]].T

            # Place eigenvector on appropriate row of transform matrix
            self.ctx.transform_matrix[i, :] = eig_vec
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_dataset_vars(self):

        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.dsvars', DIM_MEANS=self.ctx.dim_means,
                 COV_MATRIX=self.ctx.cov_matrix, TRANSFORM_MATRIX=self.ctx.transform_matrix)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("Loading dataset variables from ", self.ctx.path)
        with np.load(os.path.join(self.ctx.path, '') + self.ctx.fname + '.dsvars.npz') as data:
            self.ctx.dim_means = data['DIM_MEANS']
            self.ctx.cov_matrix = data['COV_MATRIX']
            self.ctx.transform_matrix = data['TRANSFORM_MATRIX']
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        
        # if pipe_state != None:
        #     print('PIPELINE ELEMENT DataSet : Incoming Pipe State -> ', pipe_state)

        if self.ctx.mode in ('F', 'B'):
            self._initialize()
            self._calc_dim_means()
            self._calc_cov_matrix()
            self._calc_transform_matrix()
            self._save_dataset_vars()   # Save dim_means/cov_matrix/transform_matrix to a file
            return {"created": ("DIM_MEANS", "COV_MATRIX", "TRANSFORM_MATRIX")}      
        else:
            self._initialize()            
            return {"instantiated": ("DataSet")}                    
    # ----------------------------------------------------------------------------------------------------------------------------------------