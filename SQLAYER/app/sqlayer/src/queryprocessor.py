import numpy as np
import os
import time
import timeit
from datetime import datetime
import csv
import json
import time

from sqlayer import DataManager as DataManager

class QueryProcessor:

    MAX_UINT8       = 255
    # K_READ_FACTOR   = 2

    def __init__(self, payload):
        
        # Paramaters
        self.payload            = payload
        
        # Other
        self.dmg_params         = None
        self.qp_params          = None            
        self.dmg : DataManager  = None
        self.partition_id       = None
        self.query_ref          = None
        self.query              = None
        self.num_candidates     = None
        self.candidates         = None
        self.predicate_set      = None
        self.start_time         = time.perf_counter()
        self.end_time           = None

        # Re-initialized at the start of phase 1 for each query.
        self.ANS                = None
        self.UP                 = None
        self.V                  = None
        self.L                  = None
        self.U                  = None
        self.S1                 = None
        self.S2                 = None
        self.H                  = None
        
        # Batch query variables
        self.batch_pos          = None
        self.query_start_time   = None
        self.query_end_time     = None
        self.query_result_list  = []
        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def initialize(self):
        

        # Unpack Payload
        self.dmg_params         = self.payload["dmg_params"]
        self.qp_params          = self.payload["qp_params"]
        
        # Instantite Datamanager and load relevant data
        self.dmg : DataManager      = DataManager(params=self.dmg_params)
        
        # # Temporary fix only - when running in AWS won't be required
        # self.dmg.unset_qp_globals()        
        self.dmg.load_data()
        
        self.first_stage    = np.zeros(len(self.qp_params), dtype=np.uint32)
        self.second_stage   = np.zeros(len(self.qp_params), dtype=np.uint32)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def initialize_for_query(self, qdata):        
        
        self.partition_id       = int(qdata["partition_id"])
        self.batch_pos          = int(qdata["batch_pos"])
        self.query_ref          = int(qdata["query_ref"])
        self.num_candidates     = int(qdata["num_candidates"])
        self.query              = np.array(qdata["query"], dtype=np.float32)
        self.predicate_set      = np.array(qdata["predicate_set"], dtype='U10')        
        self.candidates         = np.array(qdata["candidates"],dtype=np.uint8)

        # Unpack payload items into dmg instance variables
        cands = np.unpackbits(self.candidates)[0:self.dmg.num_vectors]
        self.dmg.candidates         = np.where(cands == 1)[0]
        self.dmg.candidate_count    = self.num_candidates
        self.dmg.Q                  = np.zeros((1,self.dmg.num_dimensions), dtype=np.float32)
        self.dmg.Q_raw              = np.zeros((1,self.dmg.num_dimensions), dtype=np.float32)
        self.dmg.Q[0,:]             = np.array(self.query, dtype=np.float32)
        self.dmg.Q_raw[0,:]         = np.array(self.query, dtype=np.float32)
        self.dmg.num_queries        = 1
        self.dmg.predicate_sets = np.atleast_2d(self.predicate_set)
        
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def recheck_predicates(self, vec_id, q_id) -> bool:
        
        vec_attrs = self.dmg.read_vector_attributes(vector_id=vec_id, type='raw')
        query_pred_set = self.dmg.predicate_sets[q_id]
        
        # Compare
        for i in range(0, len(query_pred_set), 4):
            pred = query_pred_set[i:i+4]
            attr_id = int(pred[0])
            # if self.filters[q_id, attr_id] == 0:
            #     continue
            operator = pred[1]
            if operator != '':
                operand1 = np.float32(pred[2])
                if len(pred[3]) > 0:
                    operand2 = np.float32(pred[3])
                else:
                    operand2 = None
                if operator == '<': 
                    if vec_attrs[attr_id] >= operand1:
                        print('** recheck_predicates -> Mistmatch: Vector ', str(vec_id).rjust(8,' '), 'Attribute ', attr_id, ' Predicate ', pred, ' Attribute Value: ', vec_attrs[attr_id], ' **')
                        return False
                elif operator == '<=':
                    if vec_attrs[attr_id] > operand1:
                        print('** recheck_predicates -> Mistmatch: Vector ', str(vec_id).rjust(8,' '), 'Attribute ', attr_id, ' Predicate ', pred, ' Attribute Value: ', vec_attrs[attr_id], ' **')                        
                        return False    
                elif operator == 'B': # > op1, <= op2
                    if ((vec_attrs[attr_id] <= operand1) or (vec_attrs[attr_id] > operand2) ):
                        print('** recheck_predicates -> Mistmatch: Vector: ', str(vec_id).rjust(8,' '), 'Attribute ', attr_id, ' Predicate ', pred, ' Attribute Value: ', vec_attrs[attr_id], ' **')
                        return False
                elif operator == '>':
                    if vec_attrs[attr_id] <= operand1:
                        print('** recheck_predicates -> Mistmatch: Vector ', str(vec_id).rjust(8,' '), 'Attribute ', attr_id, ' Predicate ', pred, ' Attribute Value: ', vec_attrs[attr_id], ' **')
                        return False
                elif operator == '>=':
                    if vec_attrs[attr_id] < operand1:
                        print('** recheck_predicates -> Mistmatch: Vector ', str(vec_id).rjust(8,' '), 'Attribute ', attr_id, ' Predicate ', pred, ' Attribute Value: ', vec_attrs[attr_id], ' **')
                        return False   

        # All attributes OK for this vector
        return True    
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Binary quantization preprocessing, only run once.
    def binary_quantization_preproc(self):
        # Standardize and binary quantize the queries file
        Q_std = np.divide(np.subtract(self.dmg.Q, self.dmg.tf_dim_means), self.dmg.tf_stdevs)
        self.dmg.BQQ = np.packbits(np.where(Q_std<0, 0, 1),axis=1)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Binary Quantization Hamming distance calculation and candidate selection, run once per query.
    def calc_binary_quantization_candidates(self, query_idx):
        
        # NOTE: FIXED At the moment this is a bit inefficient, because counting the bits for the Hamming distances is done in a loop
        #       Numpy 2.0 includes a bitwise_count ufunc which should allow this to be done via a broadcast

        if self.dmg.candidate_count > 0:        # Attribute Filtering has taken place
            BSET = self.dmg.bqdata[self.dmg.candidates]
            self.dmg.hammings = np.zeros(self.dmg.candidate_count, dtype=np.uint16)
        else:
            BSET = self.dmg.bqdata
            self.dmg.hammings = np.zeros(self.dmg.num_vectors,dtype=np.uint16)   # Caters for distances up to 65536

        # Hamming Distance calc
        hd_reft = timeit.default_timer()
        self.dmg.bqq = self.dmg.BQQ[query_idx,:]
        xors = np.bitwise_xor(BSET, self.dmg.bqq)        
        
        # for j, xor in enumerate(xors):
        #     self.dmg.hammings[j] = bitarray(endian='big', buffer=xor).count(1)
        
        # Numpy 2.0 version
        self.dmg.hammings = np.bitwise_count(xors).sum(axis=1)
            
        # Set a cutoff for the Binary Quantization - controls how many sorted hammings are send forward for LB calculation
        # Base this on the number of hammings done and the BQ Cutoff Percentage parameter
        bq_cutoff = np.uint64(np.divide(np.dot(self.dmg.hammings.shape[0],self.dmg.bq_cutoff_perc),100))
        bq_cutoff = np.uint16(np.maximum(bq_cutoff, np.dot((self.dmg.query_k * self.dmg.k_factor), self.dmg.bq_cutoff_perc)))
        
        # if self.dmg.candidates is None:
        if self.dmg.candidate_count == 0:
            self.dmg.candidates = np.argsort(self.dmg.hammings,axis=0)[0:bq_cutoff]
            self.dmg.candidate_hammings = self.dmg.hammings[self.dmg.candidates]
        else:
            self.dmg.candidates = self.dmg.candidates[np.argsort(self.dmg.hammings,axis=0)][0:bq_cutoff]
            self.dmg.candidate_hammings = self.dmg.hammings[np.argsort(self.dmg.hammings,axis=0)][0:bq_cutoff]
        self.dmg.candidate_count = self.dmg.candidates.shape[0]

        last_cnt = -1 * self.dmg.candidate_count
        # print('Last ', self.dmg.candidate_count, ' Hamming Distances           : ', self.dmg.candidate_hammings[last_cnt:])
            
        msg = 'Query : ' + str(query_idx) + ' BQ Search Duration'
        self.dmg.debug_timer('QueryProcessor.run_queries_approx',hd_reft, msg)                          
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # Res file size (bytes) = word_size * 2 (i.e. V, ANS) * self.dmg.query_k * self.num_queries.
    def write_res_info(self): # Called once per query. 
    
        # Open res file (mode append)
        # self._open_file('res', 'ab')
        with open(self.dmg.full_res_fname, mode='ab') as f:

            # Write self.V and self.ANS (refreshed per-query.)
            for i in range(self.dmg.query_k):
                f.write(np.uint32(self.V[i])) # uint32
                f.write(np.float32(self.ANS[i])) # float32
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def print_res_info(self, query_idx):

        # Display results
        print()
        print("********************")
        print("Results for Query ", query_idx)
        print("********************")
        print("V")
        print(self.V[0:self.dmg.query_k])
        print("ANS")
        print(self.ANS[0:self.dmg.query_k])

        if self.dmg.binary_quantization:
            print("Hammings")

            # print(self.dmg.hammings[np.uint64(self.V)])
            # print(self.dmg.hammings[np.uint64(self.V[self.dmg.candidates])])
            # Find the index positions of V values in candidates, and get index positions
            # inds = np.uint64(np.where(np.in1d(self.dmg.candidates, self.V))[0])
            # print(self.dmg.hammings[self.V[inds]])

            # print(self.dmg.candidate_hammings[0:self.dmg.query_k])            
            print(self.H[0:self.dmg.query_k])                        
            print('Best Hammings')
            print(self.dmg.candidate_hammings[0:self.dmg.query_k])
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def write_metrics_info(self): # Write metrics info for all queries.

        with open(self.dmg.full_metrics_fname, 'wb') as f:

            # Write first stage and second stage
            f.write(self.first_stage) # (num_queries, 1), uint32
            f.write(self.second_stage) # (num_queries, 1), uint32
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def print_metrics_info(self):

        print()
        print("Overall Results for Run")
        print("First Stage : ")
        print(self.first_stage, ' SUM : ', np.sum(self.first_stage), ' MEAN : ', np.mean(self.first_stage))
        print("Second Stage : ")
        print(self.second_stage, ' SUM : ', np.sum(self.second_stage), ' MEAN : ', np.mean(self.second_stage))
        print()
        print(" <---- END OF VAQPLUS PROCESSING RUN ---->")
        print()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def run_phase_one(self, query_idx):

        self.dmg.q = self.dmg.Q[query_idx,:]
        if self.dmg.adc_or_sdc == 'sdc':
            self.dmg.qq = self.dmg.QQ[query_idx,:]

        # self.ANS = np.ones((self.dmg.query_k * QueryProcessor.K_READ_FACTOR), dtype=np.float32)*np.inf
        # self.UP = np.ones((self.dmg.query_k * QueryProcessor.K_READ_FACTOR), dtype=np.float32)*np.inf
        # self.V = np.ones((self.dmg.query_k * QueryProcessor.K_READ_FACTOR), dtype=np.uint32)*np.inf

        self.ANS = np.ones((np.uint16(self.dmg.query_k * self.dmg.k_factor)), dtype=np.float32)*np.inf
        self.UP = np.ones_like(self.ANS)*np.inf
        self.V = np.ones_like(self.ANS)*np.inf
        self.H = np.zeros_like(self.ANS)*np.inf

        if self.dmg.adc_or_sdc == 'sdc':
            self.calc_sdc_distances_columnar(query_idx)
        else:
            self.calc_adc_distances_columnar(query_idx)

        # If Approx, need to set self.V and self.ANS now as we aren't doing phase two.
        # Cater for Approx-BQLB and Approx-LB. 
        # If doing BQ, will have candidates and self.L will have already been trimmed.
        # If not doing BQ, select top-K LBs from entire dataset.
        if self.dmg.precision in ('approx_lb', 'approx_bqlb'):
            # lb_idxs = np.argsort(self.L, axis=0)[0:self.dmg.query_k * QueryProcessor.K_READ_FACTOR]
            lb_idxs = np.argsort(self.L, axis=0)[0:int(self.dmg.query_k * self.dmg.k_factor)]
            ln = lb_idxs.shape[0]

            if self.dmg.candidate_count > 0:        # Either or both of attribute filtering and binary quantization have taken place
                self.V[0:ln] = self.dmg.candidates[lb_idxs]
                if self.dmg.candidate_hammings is not None:
                    self.H[0:ln] = self.dmg.candidate_hammings[lb_idxs]
                self.ANS[0:ln] = self.L[lb_idxs]
            else:
                self.V[0:ln] = lb_idxs
                self.ANS[0:ln] = self.L[lb_idxs]   
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def run_phase_two(self, query_idx):
        
        # Sort L (lower bounds) into [LL, J], also need original indices 
        J = np.argsort(self.L,axis=0)
        LL = np.sort(self.L,axis=0)

        # If Attribute Filtering has occurred, J represents indices from the cutdown RSET. Also need the real vector IDs
        if self.dmg.candidate_count > 0:
            J_FULL = self.dmg.candidates[J]
        else:
            J_FULL = J

        # Each random read on transformed file should seek to start of the info for current record, then read num_dimensions words.
        num_words_random_read = self.dmg.num_dimensions

        # Loop over all vectors; is this sensible; don't we just want to consider candidates only in terms of their LBs?
        vectors_considered_p2 = 0

        get_max_next_time = True
        if self.dmg.candidate_count > 0:
            cnt = self.dmg.candidate_count
        else:
            cnt = self.dmg.num_vectors
            
        self.dmg.ts_open_file()
        for i in range(cnt):
            if get_max_next_time:
                max_dist = self.ANS.max()
                max_dist_idx = np.argmax(self.ANS)
            
            # Ensure vector attributes meet predicate - should be rare (but possible) at this stage that there is a mismatch
            if self.dmg.candidate_count > 0:
                id = self.dmg.candidates[i]
                id = J_FULL[i]
                if not self.recheck_predicates(vec_id=id, q_id=query_idx):
                    continue
            else:
                id = i
            
            if LL[i] > max_dist:
                break
            else:
                # Random read (of num_dimensions words) from transformed file. 
                start_offset = np.uint64(J_FULL[i]*self.dmg.num_dimensions*self.dmg.word_size)
                TSET = self.dmg.tf_random_read(start_offset, num_words_random_read) # (1, num_dimensions)

                vec_dist = np.sqrt(np.sum(np.square(self.dmg.q - TSET)))
                if vec_dist <= max_dist:
                    self.ANS[max_dist_idx] = vec_dist
                    self.V[max_dist_idx] = J_FULL[i]
                    get_max_next_time = True
                else:
                    get_max_next_time = False
                vectors_considered_p2 += 1

        self.dmg.ts_close_file()
        idxs = np.argsort(self.ANS)
        ln = idxs.shape[0]
        self.ANS[0:ln] = self.ANS[idxs]
        self.V[0:ln] = self.V[idxs]

        # self.second_stage[query_idx] = vectors_considered_p2
        self.second_stage[self.batch_pos] = vectors_considered_p2
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def calc_adc_distances_columnar(self,query_idx):
        
        # Set up VAQIndex generator 
        if self.dmg.inmem_vaqdata == 'inmem_columnar':
            vaq_gene = self.dmg.generate_vaq_block_mem()
        else:
            vaq_gene = self.dmg.generate_vaq_block()
        
        # Calc ADC lookup. Reverted to legacy version to avoid the need for scipy.
        calcdists_reft = timeit.default_timer()
        
        # SCIPY VERSION
        # D = np.square(np.subtract(self.dmg.boundary_vals, self.dmg.q[None, :].ravel(), where=self.dmg.boundary_vals!=0))
        # D_MIN = minimum_filter1d(D, size=2, axis=0, origin=-1)
        # D_MAX = maximum_filter1d(D, size=2, axis=0, origin=-1)
        
        # LEGACY VERSION
        boundary_vals_wrapped = np.roll(self.dmg.boundary_vals,-1,0) # Roll rows; 1st row becomes 0th, 0th becomes last.
        D1 = np.abs(np.subtract(boundary_vals_wrapped, self.dmg.q[:,None].ravel(), where=boundary_vals_wrapped!=0, out=np.zeros_like(boundary_vals_wrapped) )  ) # cset+1
        D2 = np.abs(np.subtract(self.dmg.boundary_vals, self.dmg.q[:,None].ravel(), where=self.dmg.boundary_vals!=0, out=np.zeros_like(self.dmg.boundary_vals) )  ) # cset
        D_MIN = np.square(np.minimum(D1,D2))
        D_MAX = np.square(np.maximum(D1,D2))           
        
        msg = 'Query : ' + str(self.query_ref) + ' Build ADC Lookup'
        self.dmg.debug_timer('QueryProcessor.run_phase_one',calcdists_reft, msg, 1)
        
        dimloop_reft = timeit.default_timer()

        block_count = 0 
        for CSET in vaq_gene: # cset is a block of VAQ -> (num_vectors, 1)

            cells_for_dim = self.dmg.cells[block_count]
            qj = self.dmg.q[block_count]

            # The np.where selects RIGHT boundaries, but target_cells will be set to values 1 lower than the corresponding boundary_vals,
            # because we're only searching over [1:cells_for_dim+1] i.e. starting at row 1. So these are really LEFT boundaries.
            target_cells = np.where(qj <= self.dmg.boundary_vals[1:cells_for_dim+1, block_count]) 
            
            if target_cells[0].size == 0:
                R = cells_for_dim # If qj > all boundary_vals, put in final cell -> this is the LEFT boundary of final cell.
            else:
                R = np.min(target_cells[0]) 
  
            self.S1 = D_MIN[CSET,block_count]
            self.S2 = D_MAX[CSET,block_count]
            x = np.logical_not(CSET == R).astype(np.float32)

            # Calculate L (lower bound): L=L+x.*S1;    L is (num_vectors, 1).
            # Adds the lower bound distance for the dimension in question, to a running total which becomes the overall (squared) lower bound distance. 
            # Mask x ensures that points in the same interval along a given dimension have LB distance 0 over that dimension.
            if self.L is not None:
                self.L = self.L + np.multiply(x, self.S1)
            else:
                self.L = np.multiply(x, self.S1)

            if self.U is not None:
                self.U = self.U + self.S2
            else:
                self.U = self.S2
            block_count += 1

        # End block loop
        self.L = np.sqrt(self.L)
        self.U = np.sqrt(self.U)
        
        # # Calculate mid-bounds. Easier to retain use of self.L to avoid many changes elsewhere.
        # min_val = np.min(self.L)
        # min_dim = np.argmin(self.L)
        # self.M = np.divide(np.subtract(self.U,self.L),2)
        # self.L = self.L + self.M
        # if min_val == 0:
        #     self.L[min_dim] = 0

        msg = 'Query : ' + str(query_idx) + ' Dimensions block'        
        self.dmg.debug_timer('QueryProcessor.run_phase_one->_calc_adc_distances_columnar',dimloop_reft, msg, 1)
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    def calc_sdc_distances_columnar(self,query_idx):
        
        dimloop_reft = timeit.default_timer()
        
        # Set up VAQIndex generator (one block = one dimension)
        if self.dmg.inmem_vaqdata == 'inmem_columnar':
            vaq_gene = self.dmg.generate_vaq_block_mem()
        else:
            vaq_gene = self.dmg.generate_vaq_block()

        block_count = 0
        max_cell_count = np.uint64(np.max(self.dmg.cells))
        for CSET in vaq_gene:

            cells_for_dim = self.dmg.cells[block_count]
            qj = self.dmg.qq[block_count]

            inds = np.add(np.dot(np.uint64(qj),max_cell_count),np.uint64(CSET))

            # Use cset values to index into the distances array created for the query
            self.S1 = self.dmg.sdc_lookup_lower[inds,block_count]
            self.S2 = self.dmg.sdc_lookup_upper[inds,block_count]
            if self.L is not None:
                self.L = np.add(self.L, self.S1)
            else:
                self.L = self.S1
            if self.U is not None:
                self.U = np.add(self.U, self.S2)
            else:
                self.U = self.S2
            block_count += 1

        # End block loop
        self.L = np.sqrt(self.L)
        self.U = np.sqrt(self.U)
        
        # # Calculate mid-bounds. Easier to retain use of self.L to avoid many changes elsewhere.
        # min_val = np.min(self.L)
        # min_dim = np.argmin(self.L)
        # self.M = np.divide(np.subtract(self.U,self.L),2)
        # self.L = self.L + self.M
        # if min_val == 0:
        #     self.L[min_dim] = 0      

        msg = 'Query : ' + str(query_idx) + ' Dimensions block'        
        self.dmg.debug_timer('QueryProcessor.run_phase_one->calc_sdc_distances_columnar',dimloop_reft, msg, 1)          
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    def fine_tune_results(self, qid):
        
        # For current results, calc actual distances
        self.dmg.ds_open_file()
        for ind in range(self.V.shape[0]):
            if self.V[ind] != np.inf:
                vec = self.dmg.ds_random_read_vector(int(self.V[ind]))
                self.ANS[ind] = np.sqrt(np.sum(np.square(np.subtract(vec, self.dmg.Q_raw[qid]))))
            else:
                self.ANS[ind] = np.inf
        self.dmg.ds_close_file()
        self.V      = self.V[np.argsort(self.ANS)]
        self.H      = self.H[np.argsort(self.ANS)]              
        self.ANS    = np.sort(self.ANS)        
            
        # Check attributes for these vectors exactly satisfy predicates   
        if self.dmg.num_attributes > 0:         
            pred_matches = np.zeros((self.V.shape[0]), dtype=np.uint8)    
            for ind, vec in enumerate(self.V):
                pred_matches[ind] = self.recheck_predicates(vec,qid)
            
            fails_to_keep   = max((self.dmg.query_k - int(pred_matches.sum())),0)
            temp_V          = np.where(pred_matches==1, self.V, np.inf)
            temp_H          = np.where(pred_matches==1, self.H, np.inf)            
            temp_ANS        = np.where(pred_matches==1, self.ANS, np.inf)    

            ind = 0
            while (fails_to_keep > 0) and (ind < temp_V.shape[0]):
                if temp_V[ind]      == np.inf:
                    temp_V[ind]     = self.V[ind]
                    temp_H[ind]     = self.H[ind]                    
                    temp_ANS[ind]   = self.ANS[ind]
                    fails_to_keep   -= 1
                ind += 1

            self.V      = temp_V[np.argsort(temp_ANS)]
            self.H      = temp_H[np.argsort(temp_ANS)]                
            self.ANS    = np.sort(temp_ANS)
        dummy=0
    # ----------------------------------------------------------------------------------------------------------------------------------------        
    def run_queries_exact(self):
        
        for i in range(len(self.qp_params)):
            
            self.query_start_time = time.perf_counter()
            
            # Per-query initialization and setup
            self.initialize_for_query(json.loads(self.qp_params[i]))
            self.dmg.transform_queries()
            if self.dmg.adc_or_sdc == 'sdc':
                self.dmg.quantize_queries()      
            
            print()
            print("****************************************")
            print("Now processing query idx: ", str(self.query_ref))
            print("****************************************")            
            qreft = timeit.default_timer()

            # Reset variables
            self.ANS                            = None
            self.UP                             = None
            self.V                              = None
            self.L                              = None
            self.U                              = None
            self.S1                             = None
            self.S2                             = None
            
            self.dmg.hammings                   = None
            self.dmg.candidate_hammings         = None            

            # Phase one
            reft = timeit.default_timer()
            self.run_phase_one(0)   # Can support both inmem_columnar and disk
            msg = 'Query : ' + str(self.query_ref) + ' Phase 1 duration'
            self.dmg.debug_timer('QueryProcessor.run_queries',reft, msg)
            
            # Phase two
            reft = timeit.default_timer()
            self.run_phase_two(0)
            msg = 'Query : ' + str(self.query_ref) + ' Phase 2 duration'            
            self.dmg.debug_timer('QueryProcessor.run_queries_exact', reft, msg)
            
            msg = 'Query : ' + str(self.query_ref) + ' Query duration'            
            self.dmg.debug_timer('QueryProcessor.run_queries_exact', qreft, msg)

            # Replace and inf values with 0s
            self.V   = np.nan_to_num(self.V,posinf=0)
            self.ANS = np.nan_to_num(self.ANS,posinf=0)

            # Save query results
            self.save_query_result()

            # Print results of current query
            self.print_res_info(self.query_ref)

        # Print overall metrics for query run (all queries)
        self.print_metrics_info()
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def run_queries_approx(self):
        
        for i in range(len(self.qp_params)):
            
            self.query_start_time = time.perf_counter()
            
            # Per-query initialization and setup
            self.initialize_for_query(json.loads(self.qp_params[i]))
            self.dmg.transform_queries()
            if self.dmg.adc_or_sdc == 'sdc':
                self.dmg.quantize_queries()      
                
            # If BQ, do preprocessing: BQ query, init candidate_count, candidates, hammings variables
            if self.dmg.binary_quantization:
                self.binary_quantization_preproc()                       
            
            qreft = timeit.default_timer()            
            
            print()
            print("****************************************")
            print("Now processing query idx: ", str(self.query_ref))
            print("****************************************")
            
            # Reset variables
            self.ANS                            = None
            self.UP                             = None
            self.V                              = None
            self.L                              = None
            self.U                              = None
            self.S1                             = None
            self.S2                             = None
            self.dmg.hammings                   = None
            self.dmg.candidate_hammings         = None

            # Calculates Hamming distance and sets self.dmg.candidates ahead of phase 1
            if self.dmg.binary_quantization:
                self.calc_binary_quantization_candidates(0)            
                
            reft = timeit.default_timer()        

            # Phase one
            self.run_phase_one(0) # SDC and BQ not yet suported
            
            if self.dmg.fine_tune:
                self.fine_tune_results(0)

            msg = 'Query : ' + str(self.query_ref) + ' Phase 1 duration'
            self.dmg.debug_timer('QueryProcessor.run_queries_approx',reft, msg)
            
            msg = 'Query : ' + str(self.query_ref) + ' Query duration'            
            self.dmg.debug_timer('QueryProcessor.run_queries_approx', qreft, msg)            

            # Save query results
            self.save_query_result()

            # Print results of current query
            self.print_res_info(self.query_ref)
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    def save_query_result(self):
        # if self.dmg.candidate_hammings is None:
        #     Hammings = []
        # else:
        #     Hammings = self.dmg.candidate_hammings[0:self.dmg.query_k].tolist()
        
        if self.dmg.candidate_hammings is None:
            Hammings = []
        else:
            Hammings = self.H[0:self.dmg.query_k].astype(np.uint16).tolist()

        self.query_end_time = time.perf_counter()
        query_elapsed = self.query_end_time - self.query_start_time
        print('Query Elapsed : ', query_elapsed)
    
        query_result = {    "batch_pos"     : str(self.batch_pos),
                            "query_ref"     : str(self.query_ref),
                            "V"             : self.V[0:self.dmg.query_k].astype(np.uint32).tolist(),
                            "ANS"           : self.ANS[0:self.dmg.query_k].tolist(),
                            "H"             : Hammings
                       }    
        self.query_result_list.append(query_result)
        # self.query_result_list.append(json.dumps(query_result))
    # ----------------------------------------------------------------------------------------------------------------------------------------                        
    def build_qp_response(self):

        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        print('QP Elapsed : ', elapsed)

        qp_results  = { "partition_id"  	: str(self.partition_id),
                        "warm"		        : self.dmg.warm_start,
                        "elapsed"		    : str(elapsed),
                        "efs_bytes_read"    : str(self.dmg.efs_bytes_read),
                        "s3_gets"           : str(self.dmg.s3_gets)                        
                      }        

        qp_response = { "qp_results" 		: qp_results,
                        "query_result_list"	: self.query_result_list
                      }		
                      
        out_str = json.dumps(qp_response)
        print("In build_qp_response - json.dumps(qp_response): ", out_str)
       
        return out_str
    # ----------------------------------------------------------------------------------------------------------------------------------------                        
    def process(self):
        
        QPSession_start_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("QP Session Start Time : ", str(current_time))
        print()
        
        self.initialize()
        if self.dmg.precision == 'exact':
            self.run_queries_exact()
        else:
            self.run_queries_approx()
            
        response = self.build_qp_response()
        
        QPSession_end_time = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("QueryProcessor Session End Time : ", current_time, " Elapsed : ", str(QPSession_end_time - QPSession_start_time) )
        print()
        
        return response
    # ----------------------------------------------------------------------------------------------------------------------------------------