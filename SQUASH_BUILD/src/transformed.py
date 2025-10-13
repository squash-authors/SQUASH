import numpy as np
import os

from base import PipelineElement, TransformationSummary
from qsession import QSession

class TransformedDataSet(PipelineElement):

    def __init__(self, ctx: QSession = None):
        self.ctx                = ctx
        self.full_tf_fname      = None
        self.full_tp_fname      = None

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _initialize(self):

        np.set_printoptions(suppress=True)
        
        # Load DataSet means, covariance and transform matrices
        self._load_dataset_vars()
        
        # Load tf vars if in modes Q or P
        if self.ctx.mode in ('Q','P'):
            self._load_tf_vars()

        # Initialise variables. Note that "context" variables in the parent class/object are accessed/set, as well as local properties (filenames)
        self.full_tf_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.tf'
        self.full_tp_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.tp'
        
        # New file - standardized transformed file, only used for binary quantization.
        self.full_tfbq_fname = os.path.join(self.ctx.path, '') + self.ctx.fname + '.tfbq'
        
        # Calculate new properties for number of words/vectors per block in transformed dataset. There are fewer words in this dataset,
        # since identifiers have been removed.
        total_tf_file_words = (self.ctx.num_vectors * (self.ctx.num_dimensions))
        print("total_tf_file_words: ", str(total_tf_file_words))

        assert (total_tf_file_words % self.ctx.num_blocks == 0) and (total_tf_file_words // self.ctx.num_blocks) % self.ctx.num_dimensions == 0, "Incorrect number of blocks specified"

        self.ctx.tf_num_words_per_block = int(total_tf_file_words / self.ctx.num_blocks)
        print("self.ctx.tf_num_words_per_block: ", str(self.ctx.tf_num_words_per_block))

        self.ctx.tf_num_vectors_per_block = int(self.ctx.tf_num_words_per_block / (self.ctx.num_dimensions))
        print("self.ctx.tf_num_vectors_per_block: ", str(self.ctx.tf_num_vectors_per_block))

        self.ctx.tp_num_words_per_block = int(total_tf_file_words / self.ctx.num_dimensions)
        print("self.ctx.tp_num_words_per_block: ", str(self.ctx.tp_num_words_per_block))        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_dataset_vars(self):

        print("In _load_dataset_vars, self.ctx.path: ", self.ctx.path)

        dataset_full_varfile = os.path.join(self.ctx.path, '') + self._find_file_by_suffix('.dsvars.npz')

        print("Loading dataset variables from ", dataset_full_varfile)
        with np.load(dataset_full_varfile) as data:
            self.ctx.dim_means = data['DIM_MEANS']
            self.ctx.cov_matrix = data['COV_MATRIX']
            self.ctx.transform_matrix = data['TRANSFORM_MATRIX']
    # ----------------------------------------------------------------------------------------------------------------------------------------
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
    def _build_tf(self):

        print("Arrived at _build_tf!")
        with open(self.full_tf_fname, mode="wb") as f:
         
            gene = self.ctx.DS.generate_dataset_block()       
            for X in gene:
                Y = np.subtract(X, self.ctx.dim_means)                
                Z = np.matmul(Y, self.ctx.transform_matrix)
                f.write(Z)

            print("Finished _build_tf!")
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def generate_tf_block(self, start_offset=0):

        if self.ctx.mode in ('F','B','R'):
            block_idx = start_offset
            with open(self.full_tf_fname, mode="rb") as f:
                while True:
                    f.seek(self.ctx.tf_num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.ctx.tf_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.ctx.tf_num_vectors_per_block, self.ctx.num_dimensions), order="C")
                        yield block
                        block_idx +=1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tf_block outside of modes B, F or R")
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _build_tp(self):

        print("Arrived at _build_tp!")
        write_count = 0
        with open(self.full_tp_fname, mode='wb') as f:
            for i in range(self.ctx.num_dimensions):
                tp_dim = np.zeros(self.ctx.num_vectors, dtype=np.float32)

                # Init generator + loop var
                gene_tf = self.generate_tf_block()
                block_count = 0

                for tf_block in gene_tf:
                    tp_dim[(block_count * self.ctx.tf_num_vectors_per_block):((block_count + 1) * self.ctx.tf_num_vectors_per_block)] = tf_block[:,i]  
                    block_count += 1

                f.write(tp_dim)
                write_count += 1

        print("Finished _build_tp!") 
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Gives (num_vectors, 1) block of tp; all data for a single dimension.
    def generate_tp_block(self, start_offset=0):

        if self.ctx.mode in ('F','B','R'):
            block_idx = start_offset
            with open(self.full_tp_fname, mode="rb") as f:
                while True:
                    f.seek(self.ctx.tp_num_words_per_block * block_idx * self.ctx.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.ctx.tp_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.ctx.num_vectors, 1), order="C")  # Order F to mirror MATLAB
                        yield block
                        block_idx += 1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tp_block outside of modes F and B.")
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def generate_tfbq_block(self, start_offset=0):

        if self.ctx.mode in ('F','B','R'):
            block_idx = start_offset
            with open(self.full_tfbq_fname, mode="rb") as f:
                while True:
                    f.seek(self.ctx.tf_num_words_per_block*block_idx*self.ctx.word_size, os.SEEK_SET)
                    block = np.fromfile(file=f, count=self.ctx.tf_num_words_per_block, dtype=np.float32)

                    if block.size > 0:
                        block = np.reshape(block, (self.ctx.tf_num_vectors_per_block, self.ctx.num_dimensions), order="C")
                        yield block
                        block_idx +=1
                    else:
                        break
        else:
            raise ValueError("Entered generate_tfbq_block outside of modes B, F or R")
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # num_words_random_read is like count in other read functions.    
    def tf_random_read(self, start_offset, num_words_random_read):  

        with open(self.full_tf_fname, mode='rb') as f:
            f.seek(start_offset, os.SEEK_SET)
            block = np.fromfile(file=f, count=num_words_random_read, dtype=np.float32)

            if block.size > 0:
                block = np.reshape(block, (1, self.ctx.num_dimensions))  # Done this way round, rather than MATLAB [DIMENSION, 1]'.

            return block
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _save_tf_vars(self):

        np.savez(os.path.join(self.ctx.path, '') + self.ctx.fname + '.tfvars', TF_DIM_MEANS=self.ctx.tf_dim_means,
                 TF_STDEVS=self.ctx.tf_stdevs)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _load_tf_vars(self):

        print("Loading dataset variables from ", self.ctx.path)
        with np.load(os.path.join(self.ctx.path, '') + self.ctx.fname + '.tfvars.npz') as data:
            self.ctx.tf_dim_means = data['TF_DIM_MEANS']
            self.ctx.tf_stdevs = data['TF_STDEVS']
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def _calc_tf_dim_means_stdevs(self):
        
        # Use tp file to calculate dim-by-dim means and stdevs of transformed data (for standardization)
        gene_tp = self.generate_tp_block()
        
        # Calc dim_means and stdevs
        block_count = 0
        for tp_block in gene_tp:
            self.ctx.tf_dim_means[:,block_count] = np.mean(tp_block)
            self.ctx.tf_stdevs[:,block_count] = np.std(tp_block)
            block_count += 1
            
        # Write out standardized transformed file
        gene_tf = self.generate_tf_block()
        
        with open(self.full_tfbq_fname, mode='wb') as f:
            for tf_block in gene_tf:
                tfbq_block =  np.divide(np.subtract(tf_block, self.ctx.tf_dim_means), self.ctx.tf_stdevs)
                f.write(tfbq_block)
            
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def build(self):

        # Build tf - Initial dataset transformed by KLT matrix (generated in dataset.py)
        self._build_tf()

        # Build tp - Transposed version of transformed dataset. New format: All values for dim 1, all values for dim 2, etc...
        self._build_tp()  
        
        # Calculate and save tf vars
        self._calc_tf_dim_means_stdevs()
        self._save_tf_vars()     
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:

        # if pipe_state != None:
        #     print('PIPELINE ELEMENT TransformedDataSet : Incoming Pipe State -> ', pipe_state)
            
        self._initialize()
        if self.ctx.mode in ('F', 'B'):
            self.build()
            return {"created": ("TF_FILE", "TP_FILE")} 
        else:
            return {"instantiated": ("TransformedDataSet")}            
    # ----------------------------------------------------------------------------------------------------------------------------------------
            
