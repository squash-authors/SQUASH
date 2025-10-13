import numpy as np
import os
import sys
import time
import timeit
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
import copy
import zlib

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import as_completed

from datamanager import DataManager
from queryprocessor import QueryProcessor
from treelauncher import TreeLauncher

class QueryAllocator:
    
    MAX_QP_PROCESSES            = 10
    # MAX_CONSEC_CACHE_HITS       = 5000

    def __init__(self, payload=None):
        
        # Parameters
        self.payload                        = payload
        
        # Others
        self.dmg_params                     = None
        self.qa_params                      = None
        self.num_partitions                 = None
        self.partitions_root                = None
        self.dmg : DataManager              = None
        self.centroid_distances             = None
        self.partition_candidates           = None
        self.partition_candidate_counts     = None
        self.all_V                          = []
        self.all_ANS                        = []
        self.all_H                          = []
        self.all_P                          = []
        self.all_QREFS                      = []
        self.cache                          = {}
        
        # Acumulator variables for batch query processing
        self.partition_queries              = []
                
        # Metrics
        self.qa_start_time                  = time.perf_counter()
        self.qa_end_time                    = None
        self.qa_elapsed                     = None
        self.qa_num_queries                 = None
        self.qa_qp_calls                    = None
        self.qa_num_warm_qps                = None
        self.qa_num_cold_qps                = None
        self.qa_recall                      = None
        self.qa_qps_elapsed                 = None
        self.qa_partition_visits            = None
        self.qa_q_persec                    = None
        self.qa_efs_bytes_read              = None
        self.qa_s3_gets                     = None
        self.qa_warm_starts                 = None
        
        self.master_qa_responses             = []
        
        self.initialize()
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    def initialize(self):
        
        pl                      = json.loads(self.payload)
        self.dmg_params         = pl["dmg_params"]
        self.qa_params          = pl["qa_params"]
        self.num_partitions     = int(self.qa_params["num_partitions"])
        self.partitions_root    = self.qa_params["partitions_root"]
        
        self.dmg = DataManager(params=self.dmg_params)
        self.treelauncher = TreeLauncher(self.dmg.bfr, self.dmg.l_max, self.dmg.level, self.dmg.allocator_id)        
        
        # Temporary fix only - when running in AWS won't be required
        self.dmg.unset_qa_globals()            
        self.dmg.load_data()
        
        self.qa_partition_visits = np.zeros(self.num_partitions, dtype=np.uint32)
        
        for p in range(self.num_partitions):
            self.partition_queries.append([])
            
        for q in range(self.dmg.query_batchsize):
            self.all_V.append([])
            self.all_ANS.append([])
            self.all_H.append([])
            self.all_P.append([])
        
        self.qa_num_queries     = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_qp_calls        = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_num_warm_qps    = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_num_cold_qps    = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_elapsed         = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        self.qa_recall          = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        self.qa_qps_elapsed     = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        self.qa_q_persec        = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        self.qa_cache_hits      = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_efs_bytes_read  = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_s3_gets         = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        self.qa_warm_starts     = np.zeros(self.dmg.num_allocators, dtype=np.uint8)        
            
    # ----------------------------------------------------------------------------------------------------------------------------------------                        
    def calc_centroid_distances(self, query_num):
        dim_distances           = np.square(np.subtract(self.dmg.partition_centroids, self.dmg.Q[query_num]))    
        self.centroid_distances = np.sqrt(np.sum(dim_distances, axis=1))
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    def build_qp_payload(self, part_id):
    
        # Change dmg params for QueryProcessor
        dmg_params                 = copy.deepcopy(self.dmg_params)
        dmg_params["path"]         = str(os.path.join(self.dmg.path, self.partitions_root, str(part_id)))
        dmg_params["mode"]         = "Q"    
        dmg_params["num_vectors"]  = str(self.dmg.partition_pops[part_id])
        
        payload = { "dmg_params" : dmg_params,
                    "qp_params"  : self.partition_queries[part_id]
                }         
        
        return json.dumps(payload)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build_qa_payload(self, node):
    
        # Change dmg params for QueryProcessor
        dmg_params                 = copy.deepcopy(self.dmg_params)
        qa_params                  = copy.deepcopy(self.qa_params)

        # Change dmg_params for next level allocator
        dmg_params["allocator_id"]  = node.id
        dmg_params["level"]         = node.level
                
        payload = { "dmg_params" : dmg_params,
                    "qa_params"  : qa_params
                }         
        
        return json.dumps(payload)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def check_cache(self, q_id) -> bool:
        # Get query global ind (vector ID)
        q_vec_id = self.dmg.query_inds[q_id]

        # If in cache, print results, write metrics, return Bool
        if q_vec_id in self.cache:
            print('=====================================================')
            print('CACHE HIT - Results in Allocator ', self.dmg.allocator_id, ' for Query ', q_id)
            print('=====================================================')
            print('V')
            # print(self.cache[q_vec_id['V']])
            print(self.cache[q_vec_id]['V'])
            print('ANS')
            # print(self.cache[q_vec_id['ANS']])
            print(self.cache[q_vec_id]['ANS'])
            print('Recall')
            # print(self.cache[q_vec_id['Recall']])
            print(self.cache[q_vec_id]['Recall'])
            
            self.qa_num_queries[self.dmg.allocator_id] += 1
            # self.dmg.gt_total_hits += self.cache[q_vec_id['Recall']] * self.dmg.query_k
            self.dmg.gt_total_hits += self.cache[q_vec_id]['Recall'] * self.dmg.query_k
            self.qa_cache_hits[self.dmg.allocator_id] += 1

            return True
        else:
            return False

        pass
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def prepare_query_batch_old(self, start_qid):
        
        # Re-initialize at start of each new batch
        # self.partition_candidates       = []    # List of np.packbits arrays
        # self.partition_candidate_counts = np.zeros(self.num_partitions, dtype=np.uint32)        
        self.partition_queries          = []
        for p in range(self.num_partitions):
            self.partition_queries.append([])        

        stop = min(self.dmg.num_queries, (start_qid + self.dmg.query_batchsize))
        batch_pos = -1
        q_id = start_qid
        issues_needed = False
        # consec_cache_hits = 0 
        # for q_id in range(start_qid, stop):
        # while (batch_pos < self.dmg.query_batchsize) and (q_id < self.dmg.num_queries) and (consec_cache_hits < QueryAllocator.MAX_CONSEC_CACHE_HITS):
        while (batch_pos < self.dmg.query_batchsize - 1) and (q_id < self.dmg.num_queries):
            batch_pos += 1

            if self.dmg.caching:
                if self.check_cache(q_id):
                    q_id += 1
                    # batch_pos += 1
                    # consec_cache_hits += 1
                    continue
                else:
                    # batch_pos += 1
                    issues_needed = True
                    if batch_pos == self.dmg.query_batchsize:
                        continue
            else:
                issues_needed = True
            
            self.partition_candidates       = []    # List of np.packbits arrays
            self.partition_candidate_counts = np.zeros(self.num_partitions, dtype=np.uint32)              
            
            self.calc_centroid_distances(query_num=q_id)
            min_distance = np.min(self.centroid_distances)
            if self.dmg.precision == 'exact':
                threshold_distance = np.inf
            else:
                threshold_distance = np.dot(min_distance, self.dmg.centroid_factor)    

            if not self.dmg.bigann:
                self.dmg.apply_attribute_filters(q_id)
            else:
                self.dmg.apply_label_filters(q_id)

            if self.dmg.candidate_count < self.dmg.query_k:
                print('Query : ', q_id, ' -> QueryAllocator: Total matching candidates across all partitions - ', self.dmg.candidate_count, ' is less than query_k - ',self.dmg.query_k)
                print('               :  Will provide ', self.dmg.candidate_count, ' matches')

            issued_candidates = 0
            partitions_processed = 0
            print('Query : ',q_id, ' -> Processing Partitions in order : ', np.argsort(self.centroid_distances))
            target_candidates = self.dmg.query_k
            for p_no in np.argsort(self.centroid_distances):
                
                if (self.centroid_distances[p_no] > threshold_distance):
                    if issued_candidates > target_candidates:
                        print('QueryAllocator : Target processing requests ', target_candidates, ' reached for Query ', q_id, ' after ', partitions_processed, ' partitions')    
                        break
                    else:
                        print('QueryAllocator : Centroid threshold distance : ', np.round(threshold_distance,2), ' exceeded for Query ', q_id, ' after ', partitions_processed, ' partitions but target candidates ', target_candidates, ' not yet reached. Continuing..')

                if (self.dmg.num_attributes > 0) or (self.dmg.bigann):
                    p_cands, self.partition_candidate_counts[p_no] = self.dmg.get_filtered_partition_vectors(p_no)
                    # self.partition_candidates.append(p_cands)
                else:
                    self.partition_candidate_counts[p_no] = self.dmg.partition_pops[p_no]

                if self.partition_candidate_counts[p_no] > 0:
                    self.partition_candidates.append(p_cands)
                else:
                    continue                    

                # Cater for 0 attribute scenario
                if self.partition_candidate_counts[p_no] == self.dmg.partition_pops[p_no]:
                    p_num_candidates    = 0
                    p_predicate_set     = []       
                    p_candidates        = []
                else:
                    p_num_candidates    = str(self.partition_candidate_counts[p_no])
                    if not self.dmg.bigann:
                        p_predicate_set     = self.dmg.predicate_sets[q_id].tolist()
                    # p_candidates        =  self.partition_candidates[p_no].tolist()         

                    if self.dmg.use_compression:
                        p_candidates        =  str(zlib.compress(self.partition_candidates[partitions_processed], level=3))
                    else:    
                        p_candidates        =  self.partition_candidates[partitions_processed].tolist()                      

                if self.partition_candidate_counts[p_no] > 0:
                    if not self.dmg.bigann:
                        partition_querydata = { "partition_id"      : str(p_no),
                                                "batch_pos"         : str(batch_pos),
                                                "query_ref"         : str(q_id),
                                                "num_candidates"    : p_num_candidates,
                                                "query"             : self.dmg.Q_raw[q_id].tolist(),
                                                "predicate_set"     : p_predicate_set,
                                                "candidates"        : p_candidates
                                            }
                    else:
                        partition_querydata = { "partition_id"      : str(p_no),
                                                "batch_pos"         : str(batch_pos),
                                                "query_ref"         : str(q_id),
                                                "num_candidates"    : p_num_candidates,
                                                "query"             : self.dmg.Q_raw[q_id].tolist(),
                                                "candidates"        : p_candidates
                                            }                        
                    
                    # self.partition_queries[p_no].append(partition_querydata)
                    self.partition_queries[p_no].append(json.dumps(partition_querydata))
                    # issued_candidates       += int(p_num_candidates)
                    issued_candidates       += int(self.partition_candidate_counts[p_no])                    
                    partitions_processed    += 1      
                
            q_id += 1
        
        return issues_needed, q_id
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def prepare_query_batch(self, start_qid):
        
        # Re-initialize at start of each new batch
        self.partition_queries          = []
        for p in range(self.num_partitions):
            self.partition_queries.append([])        

        stop = min(self.dmg.num_queries, (start_qid + self.dmg.query_batchsize))
        batch_pos = -1
        q_id = start_qid
        issues_needed = False

        while (batch_pos < self.dmg.query_batchsize - 1) and (q_id < self.dmg.num_queries):
            batch_pos += 1

            if self.dmg.caching:
                if self.check_cache(q_id):
                    q_id += 1
                    # batch_pos += 1
                    # consec_cache_hits += 1
                    continue
                else:
                    # batch_pos += 1
                    issues_needed = True
                    if batch_pos == self.dmg.query_batchsize:
                        continue
            else:
                issues_needed = True
            
            self.partition_candidates       = []    # List of np.packbits arrays
            self.partition_candidate_counts = np.zeros(self.num_partitions, dtype=np.uint32)              
            
            self.calc_centroid_distances(query_num=q_id)
            min_distance = np.min(self.centroid_distances)
            if self.dmg.precision == 'exact':
                threshold_distance = np.inf
            else:
                threshold_distance = np.dot(min_distance, self.dmg.centroid_factor)    

            filtering = True
            if (self.dmg.num_attributes > 0) and (not self.dmg.bigann):
                self.dmg.apply_attribute_filters(q_id)
            elif self.dmg.bigann:
                self.dmg.apply_label_filters(q_id)
            else:
                self.dmg.candidate_count = self.dmg.num_vectors
                filtering = False

            if (self.dmg.candidate_count > 0) and (self.dmg.candidate_count < self.dmg.query_k):
                print('Query : ', q_id, ' -> QueryAllocator: Total matching candidates across all partitions - ', self.dmg.candidate_count, ' is less than query_k - ',self.dmg.query_k)
                print('               :  Will provide ', self.dmg.candidate_count, ' matches')

            issued_candidates = 0
            partitions_processed = 0
            print('Query : ',q_id, ' -> Processing Partitions in order : ', np.argsort(self.centroid_distances))
            target_candidates = min(self.dmg.query_k, self.dmg.candidate_count)
            for p_no in np.argsort(self.centroid_distances):
                
                # Threshold reached and target candidates obtained - finished for this query
                if (self.centroid_distances[p_no] > threshold_distance):
                    if issued_candidates >= target_candidates:
                        print('QueryAllocator : Target processing requests ', target_candidates, ' reached for Query ', q_id, ' after ', partitions_processed, ' partitions')  
                        partitions_processed += 1  
                        break
                    else:
                        print('QueryAllocator : Centroid threshold distance : ', np.round(threshold_distance,2), ' exceeded for Query ', q_id, ' after ', partitions_processed, ' partitions but target candidates ', target_candidates, ' not yet reached. Continuing..')

                # Scenario 1: No filtering
                if not filtering:
                    self.partition_candidate_counts[p_no]   = 0
                    p_num_candidates                        = 0
                    p_predicate_set                         = []       
                    p_candidates                            = []                
                    self.partition_candidates.append(p_candidates)
                    issued_candidates += self.dmg.partition_pops[p_no]

                # Scenario 2: Filtering
                else:
                    p_cands, self.partition_candidate_counts[p_no] = self.dmg.get_filtered_partition_vectors(p_no)
                    self.partition_candidates.append(p_cands)   # NB p_cands could be empty list, but appending anyway to keep list in sync with counts
                    
                    # Scenario 3: Filtering but zero candidates in this partition. Move to next partition.
                    if self.partition_candidate_counts[p_no] == 0:
                        partitions_processed += 1
                        continue
                    else:
                        issued_candidates += int(self.partition_candidate_counts[p_no]) 

                    # Prepare querydata            
                    p_num_candidates    = str(self.partition_candidate_counts[p_no])
                    if not self.dmg.bigann:
                        p_predicate_set     = self.dmg.predicate_sets[q_id].tolist()
                    if self.dmg.use_compression:
                        p_candidates        =  str(zlib.compress(self.partition_candidates[partitions_processed], level=3))
                    else:    
                        p_candidates        =  self.partition_candidates[partitions_processed].tolist()                      

                if not self.dmg.bigann:
                    partition_querydata = { "partition_id"      : str(p_no),
                                            "batch_pos"         : str(batch_pos),
                                            "query_ref"         : str(q_id),
                                            "num_candidates"    : p_num_candidates,
                                            "query"             : self.dmg.Q_raw[q_id].tolist(),
                                            "predicate_set"     : p_predicate_set,
                                            "candidates"        : p_candidates
                                        }
                else:
                    partition_querydata = { "partition_id"      : str(p_no),
                                            "batch_pos"         : str(batch_pos),
                                            "query_ref"         : str(q_id),
                                            "num_candidates"    : p_num_candidates,
                                            "query"             : self.dmg.Q_raw[q_id].tolist(),
                                            "candidates"        : p_candidates
                                        }                        
                
                self.partition_queries[p_no].append(json.dumps(partition_querydata))
                partitions_processed += 1
                
            # Finished with this query
            q_id += 1
        
        return issues_needed, q_id
    # ----------------------------------------------------------------------------------------------------------------------------------------  
    def initialize_results(self):
        # Initialize
        self.all_V      = []
        self.all_ANS    = []
        self.all_H      = []
        self.all_P      = []       
        self.all_QREFS  = [] 
        for q in range(self.dmg.query_batchsize):
            self.all_V.append([])
            self.all_ANS.append([])
            self.all_H.append([])
            self.all_P.append([])        
        dummy=0
    # ----------------------------------------------------------------------------------------------------------------------------------------            
    def unload_qp_response(self, item):

        response            = json.loads(item)
        qp_results          = response["qp_results"]        
        query_result_list   = response["query_result_list"]
        
        partition_id        = int(qp_results["partition_id"])
        warm                = qp_results["warm"]
        elapsed             = np.float32(qp_results["elapsed"])
        efs_bytes_read      = np.uint32(qp_results["efs_bytes_read"])
        s3_gets             = np.uint32(qp_results["s3_gets"])        
        
        # Update QA metrics from QP results
        self.qa_partition_visits[partition_id]          += 1
        self.qa_qp_calls[self.dmg.allocator_id]         += 1
        self.qa_qps_elapsed[self.dmg.allocator_id]      += elapsed
        if warm:
            self.qa_num_warm_qps[self.dmg.allocator_id] += 1
        else:
            self.qa_num_cold_qps[self.dmg.allocator_id] += 1        
        self.qa_efs_bytes_read[self.dmg.allocator_id]   += efs_bytes_read
        self.qa_s3_gets[self.dmg.allocator_id]          += s3_gets
        
        # Unload query results
        for query_result in query_result_list:
        # for query_result in json.loads(query_result_list):
            batch_pos   = int(query_result["batch_pos"])
            query_ref   = int(query_result["query_ref"])
            V           = query_result["V"]
            ANS         = query_result["ANS"]
            H           = query_result["H"]
            
            temp_V      = np.array(V, dtype=np.uint32)
            # temp_V2     = np.where(self.dmg.partition_vectors[:,partition_id] == 1)[0][temp_V]      # Convert partition vecids to global vecids
            temp_V2     = self.dmg.find_partition_vecids(temp_V,partition_id)                       # Convert partition vecids to global vecids
            self.all_V[batch_pos].extend(temp_V2.tolist())
            self.all_ANS[batch_pos].extend(ANS)
            if len(H) == 0:
                self.all_H[batch_pos].extend([0] * self.dmg.query_k)
            else:
                self.all_H[batch_pos].extend(H)
            self.all_QREFS.append(query_ref)
            self.all_P[batch_pos].extend([partition_id] * self.dmg.query_k)

    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def conclude_query_batch(self):
        
        batch_qrefs = np.sort(np.unique(self.all_QREFS))
        num_results = 0

        if len(self.all_V) > 0:
            for bpos in range(self.dmg.query_batchsize):
                
                if self.all_V[bpos] == []:      # Partial batch - end of run
                    continue
                
                # q_id        = batch_qrefs[bpos]
                q_id        = batch_qrefs[num_results]
                V           = np.array(self.all_V[bpos], dtype=np.uint32)
                ANS         = np.array(self.all_ANS[bpos], dtype=np.float32)
                H           = np.array(self.all_H[bpos], dtype=np.uint16)
                P           = np.array(self.all_P[bpos], dtype=np.uint16)
                final_V     = V[np.argsort(ANS)][0:self.dmg.query_k]
                final_H     = H[np.argsort(ANS)][0:self.dmg.query_k]
                final_ANS   = np.sort(ANS)[0:self.dmg.query_k]   
                final_P     = P[np.argsort(ANS)][0:self.dmg.query_k]

                num_results += 1   
                    
                print('=====================================================')
                print('Results in Allocator ', self.dmg.allocator_id, ' for Query ', q_id)
                print('=====================================================')
                print('V')
                print(final_V)
                print('ANS')
                print(final_ANS)
                print('In Partition')
                print(final_P)
                if self.dmg.precision == 'approx_bqlb':
                    print('Hammings')
                    print(final_H)

                # Check Recall if required, update running gt hits total
                recall = 0
                if self.dmg.check_recall:
                    hits                    = np.intersect1d(self.dmg.gt_data[q_id,1:self.dmg.query_k+1], final_V[0:self.dmg.query_k]).shape[0]
                    self.dmg.gt_total_hits  += hits
                    # recall                  = hits / self.dmg.query_k   
                    recall                  = hits / min(self.dmg.query_k, final_V.shape[0])
                    print('Query ' + str(q_id) +  ' Recall@' + str(self.dmg.query_k) + ' = ' + str(recall))
                    print()            
                    
                self.qa_num_queries[self.dmg.allocator_id] += 1

                cache_entry = {
                    "V":        final_V,
                    "ANS":      final_ANS,
                    "Recall":   recall
                }
                # self.cache[self.dmg.query_inds[q_id]] = cache_entry 
                self.cache[self.dmg.query_inds[q_id]] = cache_entry   
    
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def build_qa_response(self):

        self.qa_end_time = time.perf_counter()
        self.qa_elapsed[self.dmg.allocator_id] = self.qa_end_time - self.qa_start_time
        self.qa_q_persec[self.dmg.allocator_id] = np.divide(self.dmg.num_queries, self.qa_elapsed[self.dmg.allocator_id], dtype=np.float32)
        print('QA Elapsed : ', self.qa_elapsed[self.dmg.allocator_id])
       
       # Add any EFS reads (should be 0) or s3 gets for this allocator
        self.qa_efs_bytes_read[self.dmg.allocator_id] += self.dmg.efs_bytes_read
        self.qa_s3_gets[self.dmg.allocator_id] += self.dmg.s3_gets
        
        # Flag QA warm/cold start
        self.qa_warm_starts[self.dmg.allocator_id] = self.dmg.warm_start
       
        params = {  "qa_allocator_id"       : str(self.dmg.allocator_id),
                    "qa_num_queries"        : self.qa_num_queries.tolist(),
                    "qa_recall"             : self.qa_recall.tolist(),
                    "qa_elapsed"            : self.qa_elapsed.tolist(),
                    "qa_qp_calls"           : self.qa_qp_calls.tolist(),
                    "qa_num_warm_qps"       : self.qa_num_warm_qps.tolist(),
                    "qa_num_cold_qps"       : self.qa_num_cold_qps.tolist(),
                    "qa_qps_elapsed"        : self.qa_qps_elapsed.tolist(),
                    "qa_partition_visits"   : self.qa_partition_visits.tolist(),
                    "qa_q_persec"           : self.qa_q_persec.tolist(),
                    "qa_cache_hits"         : self.qa_cache_hits.tolist(),
                    "qa_efs_bytes_read"     : self.qa_efs_bytes_read.tolist(),
                    "qa_s3_gets"            : self.qa_s3_gets.tolist(),   
                    "qa_warm_starts"        : self.qa_warm_starts.tolist()                                         
                 }
        
        # print('QA Response:')
        # print(params)
        
        return json.dumps(params)
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def build_merged_qa_response(self):

        all_qa_num_queries      = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        all_qa_recall           = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        all_qa_elapsed          = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        all_qa_qp_calls         = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        all_qa_num_warm_qps     = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        all_qa_num_cold_qps     = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        all_qa_qps_elapsed      = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        all_qa_partition_visits = np.zeros(self.num_partitions,dtype=np.uint32)
        all_qa_q_persec         = np.zeros(self.dmg.num_allocators, dtype=np.float32)
        all_qa_cache_hits       = np.zeros(self.dmg.num_allocators, dtype=np.uint32)
        all_qa_efs_bytes_read   = np.zeros(self.dmg.num_allocators, dtype=np.uint32)        
        all_qa_s3_gets          = np.zeros(self.dmg.num_allocators, dtype=np.uint32)                
        all_qa_warm_starts      = np.zeros(self.dmg.num_allocators, dtype=np.uint8)                
        
        for r in self.master_qa_responses:
            qa_res = json.loads(r)
            all_qa_num_queries      += np.array(qa_res["qa_num_queries"],       dtype=np.uint32)
            all_qa_recall           += np.array(qa_res["qa_recall"],            dtype=np.float32)
            all_qa_elapsed          += np.array(qa_res["qa_elapsed"],           dtype=np.float32)
            all_qa_qp_calls         += np.array(qa_res["qa_qp_calls"],          dtype=np.uint32)
            all_qa_num_warm_qps     += np.array(qa_res["qa_num_warm_qps"],      dtype=np.uint32)
            all_qa_num_cold_qps     += np.array(qa_res["qa_num_cold_qps"],      dtype=np.uint32)
            all_qa_qps_elapsed      += np.array(qa_res["qa_qps_elapsed"],       dtype=np.float32)
            all_qa_partition_visits += np.array(qa_res["qa_partition_visits"],  dtype=np.uint32)
            all_qa_q_persec         += np.array(qa_res["qa_q_persec"],          dtype=np.float32)
            all_qa_cache_hits       += np.array(qa_res["qa_cache_hits"],        dtype=np.uint32)
            all_qa_efs_bytes_read   += np.array(qa_res["qa_efs_bytes_read"],    dtype=np.uint32)            
            all_qa_s3_gets          += np.array(qa_res["qa_s3_gets"],           dtype=np.uint32) 
            all_qa_warm_starts      += np.array(qa_res["qa_warm_starts"],       dtype=np.uint8)                       
            
        params = {  "qa_allocator_id"       : str(self.dmg.allocator_id),
                    "qa_num_queries"        : all_qa_num_queries.tolist(),
                    "qa_recall"             : all_qa_recall.tolist(),
                    "qa_elapsed"            : all_qa_elapsed.tolist(),
                    "qa_qp_calls"           : all_qa_qp_calls.tolist(),
                    "qa_num_warm_qps"       : all_qa_num_warm_qps.tolist(),
                    "qa_num_cold_qps"       : all_qa_num_cold_qps.tolist(),
                    "qa_qps_elapsed"        : all_qa_qps_elapsed.tolist(),
                    "qa_partition_visits"   : all_qa_partition_visits.tolist(),
                    "qa_q_persec"           : all_qa_q_persec.tolist(),
                    "qa_cache_hits"         : all_qa_cache_hits.tolist(),
                    "qa_efs_bytes_read"     : all_qa_efs_bytes_read.tolist(),
                    "qa_s3_gets"            : all_qa_s3_gets.tolist(),
                    "qa_warm_starts"        : all_qa_warm_starts.tolist()                                        
                 }
        
        print('QA Response:')
        print(params)
        
        return json.dumps(params)
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    # This represents the AWS Lambda QueryProcessor              
    def query_processor(self, payload, pno, batchno):
       
        # Redirect output/error to named file under /logs
        pl      = json.loads(payload)
        # dmgp    = pl["dmg_params"]
        # qpp     = pl["qp_params"]

        stub = self.dmg.fname + "_qprocs_A" + str(self.dmg.allocator_id)
        logpath = Path('logs/' + stub)
        out = os.path.join(logpath, "B" + str(batchno) + "_P" + str(pno) + "_" + str(os.getpid()) + ".out")
        sys.stdout = open(out, "w")
        sys.stderr = sys.stdout            
        
        # Instantiate and run QueryProcessor
        queryprocessor = QueryProcessor(payload=pl )
        response = queryprocessor.process()
        
        return response
    # ---------------------------------------------------------------------------------------------------------------------------------------- 
    # This represents the AWS Lambda QueryAllocator              
    def query_allocator(self, payload):

        # take payload in, instantiate qa, qa.process
        pl = json.loads(payload)
        dmgp    = pl["dmg_params"]
        qap     = pl["qa_params"]

        stub = dmgp["fname"] + "_aprocs"
        logpath = Path('logs/' + stub)
        out = os.path.join(logpath, "A" + str(dmgp["allocator_id"]) + "_" + str(os.getpid()) + ".out")
        sys.stdout = open(out, "w")
        sys.stderr = sys.stdout      
        
        queryallocator = QueryAllocator(payload=payload)
        response = queryallocator.allocate()
        return response

    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def allocate(self):
        
        # Outer context manager
        node_gene = self.treelauncher.node_generator()        
        with ProcessPoolExecutor(max_workers=self.dmg.bfr) as alloc_executor:

            alloc_futures = []
            if (self.dmg.l_max - self.dmg.level) >= 1:
                for node in node_gene:
                    payload = self.build_qa_payload(node)
                    alloc_futures.append(alloc_executor.submit(self.query_allocator, payload=payload))
                    # res = self.query_allocator(payload=payload)
                    # self.master_qa_responses.append(res)        
        
            print()
            print("QueryAllocator ", self.dmg.allocator_id," Session Begins -> Run Mode ", self.dmg.mode)
            print("============================================")
            QASession_start_time = timeit.default_timer()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Session Start Time : ", str(current_time))
            print()        
            
            self.dmg.open_querydata_file()
            self.dmg.Q_raw = self.dmg.Q.copy()      # Use these for sending to QueryAllocators. Will need to be transformed there based on partition means/cov/transform matrices
            self.dmg.transform_queries()            # Note this uses whole datasets means/cov/transform matrices
            
            # Set up folder for MP logs
            stub = self.dmg.fname + "_qprocs_A" + str(self.dmg.allocator_id)
            logpath = Path('logs/' + stub)
            if os.path.exists(logpath):
                shutil.rmtree(logpath)
            os.mkdir(logpath)             

            # Query Loop
            if self.dmg.query_batchsize > self.dmg.num_queries:
                # print("QueryAllocator ", self.dmg.allocator_id," Requested Query Batch Size is > number of queries. Restricting to ", self.dmg.num_queries -1)
                # self.dmg.query_batchsize = self.dmg.num_queries - 1
                print("QueryAllocator ", self.dmg.allocator_id," Requested Query Batch Size is > number of queries. Restricting to ", self.dmg.num_queries)
                self.dmg.query_batchsize = self.dmg.num_queries                
            
            # Prepare the first query batch
            issues_needed, start_qid = self.prepare_query_batch(0)
            # start_qid = self.dmg.query_batchsize
            # start_qid = self.dmg.query_batchsize - 1            
                
            # Inner contect manager
            num_procs = min(QueryAllocator.MAX_QP_PROCESSES, self.num_partitions)
            with ProcessPoolExecutor(max_workers=num_procs) as executor:        

                # for start_qid in range(self.dmg.query_batchsize, self.dmg.num_queries, self.dmg.query_batchsize):
                for batchno in range (np.uint16(np.ceil(np.divide(self.dmg.num_queries, self.dmg.query_batchsize)))):
                    futures = []
                    # Issue QP Workloads
                    # self.initialize_results()
                    if issues_needed:
                        for partition_id in range(self.num_partitions):
                            if self.partition_queries[partition_id] != []:
                                payload = self.build_qp_payload(partition_id)
                                print('QA - Payload Size before submit : ', len(payload))
                                futures.append( executor.submit(self.query_processor, payload=payload, pno=partition_id, batchno=batchno) )
                                # res = self.query_processor(payload=payload, pno=partition_id, batchno=batchno)
                                # self.unload_qp_response(res)                        
                    
                    # Prepare next query batch
                    if start_qid < self.dmg.num_queries:
                        issues_needed, start_qid = self.prepare_query_batch(start_qid)
                        
                    # Collect results for previous query batch
                    self.initialize_results()
                    for future in as_completed(futures):
                        self.unload_qp_response(future.result())
                        
                    # Finish processing batch
                    self.conclude_query_batch()
                    # start_qid += self.dmg.query_batchsize 

                    # # batch_pos == -1 implies that there is no more work to be done (in prepare_query_batch)
                    # if batch_pos == -1:
                    #     break

            # End of Inner contect manager
            QASession_end_time = timeit.default_timer()
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("QueryAllocator Session End Time : ", current_time, " Elapsed : ", str(QASession_end_time - QASession_start_time) )
            print()

            if self.dmg.check_recall:
                # self.qa_recall = self.dmg.gt_total_hits / (self.dmg.num_queries * self.dmg.query_k)
                self.qa_recall[self.dmg.allocator_id] = self.dmg.gt_total_hits / (self.qa_num_queries[self.dmg.allocator_id] * self.dmg.query_k)
                print('************************************')
                print('Overall Recall@' + str(self.dmg.query_k) + ' for Session = ' + str(self.qa_recall[self.dmg.allocator_id]))
                print('************************************')
                print()        
                
            # self.master_qa_responses.append(self.build_qa_response())

            # Local allocator processing complete
            if (self.dmg.l_max - self.dmg.level) >= 1:
                for alloc_future in as_completed(alloc_futures):
                    self.master_qa_responses.append(alloc_future.result())
                    
            self.master_qa_responses.append(self.build_qa_response())                    
                
        # End of outer context manager                
    
        return self.build_merged_qa_response()                
    # ----------------------------------------------------------------------------------------------------------------------------------------      
    def prvd_processor(self):

        for vec_id in self.dmg.vecs_to_print:
  
            # # Identify partition
            # self.dmg.load_partitioner_vars()
            # partition_for_vector = np.where(self.dmg.partition_vectors[vec_id,:] == 1)[0][0]
            # # Get partition vecid
            # partition_vec_id = int(np.sum(self.dmg.partition_vectors[:,partition_for_vector][0:vec_id]))
            
            # 23/09/24 Get local vecid and partition
            self.dmg.load_partitioner_vars()
            partition_for_vector, partition_vec_id = self.dmg.find_partition_vector(vec_id)              

            # print header lines
            print("-" * 80)
            print('VECTOR ID            : ', vec_id)
            print('PARTITION            : ', partition_for_vector)
            print('PARTITION VECTOR ID  : ', partition_vec_id)
            print("-" * 80)
            print()
            
            # Instantiate and load DataManager for relevant path
            prvd_dmg_params = copy.deepcopy(self.dmg_params)
            prvd_path = str(os.path.join(prvd_dmg_params["path"], self.partitions_root, str(partition_for_vector)))
            prvd_dmg_params["path"]             = prvd_path
            prvd_dmg_params["num_vectors"]      = str(self.dmg.partition_pops[partition_for_vector])
            prvd_dmg_params["vecs_to_print"]    = [partition_vec_id]
            # prvd_dmg_params = json.dumps(prvd_dmg_params) 
            
            datamanager_prvd = DataManager(params=prvd_dmg_params)
            
            # Temporary fix only - when running in AWS won't be required
            datamanager_prvd.unset_qp_globals()            
            datamanager_prvd.load_for_prvd()
    
            # Call prvd for partition vecid
            datamanager_prvd.prvd()
    # ----------------------------------------------------------------------------------------------------------------------------------------    
     
     
     
     
