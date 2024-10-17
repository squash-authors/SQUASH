import numpy as np
import os
import sys
import timeit
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
import copy
import boto3
from io import StringIO

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

class Coordinator:
    
    def __init__(self, payload=None):
        
        # Parameters
        self.payload                = payload
        
        # Others
        self.dmg_params             = None
        self.qa_params              = None
        self.num_allocators         = None
        self.num_partitions         = None
                
        # Metrics
        self.co_start_dt            = timeit.default_timer()
        self.co_end_dt              = None
        self.co_elapsed             = 0
        self.co_num_queries         = 0
        self.co_recall              = 0
        self.co_qa_calls            = 0
        self.co_qa_qp_calls         = 0
        self.co_num_warm_qps        = 0
        self.co_num_cold_qps        = 0
        self.co_qa_elapsed          = 0
        self.co_qa_qps_elapsed       = 0
        self.co_qa_num_queries      = None
        self.co_qa_recalls          = None
        self.co_partition_visits    = None
        self.co_q_persec            = None
        self.co_qa_cache_hits       = None
        self.co_qa_efs_bytes_read   = None
        self.co_qa_s3_gets          = None
        self.co_qa_warm_starts      = None        
        self.co_metrics_fname       = None
        
        self.initialize()
    # ----------------------------------------------------------------------------------------------------------------------------------------                    
    def initialize(self):
        
        from sqlayer import QueryAllocator
        from sqlayer import TreeLauncher
        from sqlayer import GQA
        
        global gqa 
        gqa = GQA.getInstance()
        
        # pl                          = json.loads(self.payload)
        self.dmg_params             = self.payload["dmg_params"]
        self.qa_params              = self.payload["qa_params"]
        self.num_allocators         = int(self.dmg_params["num_allocators"])
        self.num_partitions         = int(self.qa_params["num_partitions"])
        self.co_qa_num_queries      = np.zeros(self.num_allocators, dtype=np.uint32)
        self.co_qa_recalls          = np.zeros(self.num_allocators, dtype=np.float32)
        self.co_partition_visits    = np.zeros(self.num_partitions, dtype=np.uint32)
        self.co_q_persec            = np.zeros(self.num_allocators, dtype=np.float32)
        self.co_qa_cache_hits       = np.zeros(self.num_allocators, dtype=np.uint32)
        self.co_qa_efs_bytes_read   = np.zeros(self.num_allocators, dtype=np.uint32)        
        self.co_qa_s3_gets          = np.zeros(self.num_allocators, dtype=np.uint32)
        self.co_qa_warm_starts      = np.zeros(self.num_allocators, dtype=np.uint8)        
        self.co_metrics_fname       = os.path.join(self.dmg_params["path"], '') + self.dmg_params["fname"] + '.comet'
        self.treelauncher           = TreeLauncher(int(self.dmg_params["bfr"]), int(self.dmg_params["l_max"]),0,-1)
    # ----------------------------------------------------------------------------------------------------------------------------------------                        
    def build_qa_payload(self, node):
        self.dmg_params["allocator_id"] = str(node.id)
        self.dmg_params["level"]        = str(node.level)
        payload = { "dmg_params" : self.dmg_params,
                    "qa_params"  : self.qa_params
                  }
        return json.dumps(payload)
    # ----------------------------------------------------------------------------------------------------------------------------------------          
    def unload_qa_response(self, item):
        # response            = json.loads(item)
        response            = json.loads(json.loads(item['Payload'].read().decode('utf-8')))
        
        qa_allocator_id     = int(response["qa_allocator_id"])
        qa_num_queries      = np.array(response["qa_num_queries"],      dtype=np.uint32)
        qa_recall           = np.array(response["qa_recall"],           dtype=np.float32)
        qa_elapsed          = np.array(response["qa_elapsed"],          dtype=np.float32)
        qa_qp_calls         = np.array(response["qa_qp_calls"],         dtype=np.uint32)
        qa_num_warm_qps     = np.array(response["qa_num_warm_qps"],     dtype=np.uint32)
        qa_num_cold_qps     = np.array(response["qa_num_cold_qps"],     dtype=np.uint32)
        qa_qps_elapsed      = np.array(response["qa_qps_elapsed"],      dtype=np.float32)
        qa_partition_visits = np.array(response["qa_partition_visits"], dtype=np.uint32)
        qa_q_persec         = np.array(response["qa_q_persec"],         dtype=np.float32)
        qa_cache_hits       = np.array(response["qa_cache_hits"],       dtype=np.uint32)
        qa_efs_bytes_read   = np.array(response["qa_efs_bytes_read"],   dtype=np.uint32)
        qa_s3_gets          = np.array(response["qa_s3_gets"],          dtype=np.uint32)
        qa_warm_starts      = np.array(response["qa_warm_starts"],      dtype=np.uint8)        
        
        # Update metrics
        self.co_num_queries         += np.sum(qa_num_queries)
        self.co_qa_num_queries      += qa_num_queries
        # self.co_qa_calls            += 1
        self.co_qa_qp_calls         += qa_qp_calls
        self.co_num_warm_qps        += qa_num_warm_qps
        self.co_num_cold_qps        += qa_num_cold_qps
        self.co_qa_elapsed          += qa_elapsed
        self.co_qa_qps_elapsed      += qa_qps_elapsed
        self.co_partition_visits    += qa_partition_visits
        self.co_q_persec            += qa_q_persec
        self.co_qa_cache_hits       += qa_cache_hits
        self.co_qa_efs_bytes_read   += qa_efs_bytes_read
        self.co_qa_s3_gets          += qa_s3_gets
        self.co_qa_warm_starts      += qa_warm_starts        
        self.co_qa_recalls          += qa_recall
        
    # ----------------------------------------------------------------------------------------------------------------------------------------          
    def build_co_response(self):                # CONVERT METRICS WRITE TO S3
        
        self.co_end_dt  = timeit.default_timer()
        self.co_elapsed = np.float32(self.co_end_dt - self.co_start_dt)
        # print('CO Elapsed : ', self.co_elapsed)

        # Calculate overall Recall
        self.co_recall = np.divide(np.sum(np.multiply(self.co_qa_recalls, self.co_qa_num_queries)),self.co_num_queries)
        # self.co_recall = np.mean(self.co_qa_recalls)
        # print('CO Recall  : ', self.co_recall)
        # print()
               
        co_params = {   "co_start_dt"           : str(self.co_start_dt),
                        "co_end_dt"             : str(self.co_end_dt),
                        "co_elapsed"            : str(self.co_elapsed),                        
                        "co_recall"             : str(self.co_recall),
                        "co_num_queries"        : str(self.co_num_queries),
                        # "co_qa_calls"           : str(self.co_qa_calls),
                        "co_qa_calls"           : str(self.treelauncher.num_allocs),
                        "co_qa_qp_calls"        : str(np.sum(self.co_qa_qp_calls)),
                        "co_num_warm_qps"       : str(np.sum(self.co_num_warm_qps)),
                        "co_num_cold_qps"       : str(np.sum(self.co_num_cold_qps)),
                        "co_qa_elapsed"         : str(np.sum(self.co_qa_elapsed)),
                        "co_qa_qps_elapsed"     : str(np.sum(self.co_qa_qps_elapsed)),
                        "co_qa_recalls"         : self.co_qa_recalls.tolist(),
                        "co_partition_visits"   : self.co_partition_visits.tolist(),
                        "co_qa_q_persecs"       : self.co_q_persec.tolist(),
                        "co_q_persec"           : str(np.sum(self.co_q_persec)),
                        "co_qa_cache_hits"      : str(np.sum(self.co_qa_cache_hits)),
                        "co_qa_efs_bytes_read"  : str(np.sum(self.co_qa_efs_bytes_read)),
                        "co_qa_s3_gets"         : str(np.sum(self.co_qa_s3_gets)),
                        "co_qa_warm_starts"     : str(np.sum(self.co_qa_warm_starts))                        
                    }
        
        self.dmg_params["allocator_id"] = str(0)
        response  = {   "dmg_params" : self.dmg_params,
                        "co_params"  : co_params
                    }
        
        print('CO Response:')
        # print(response)
        print("co_elapsed: ", str(self.co_elapsed))
        print("co_recall: ", str(self.co_recall))
        print("co_num_queries: ", str(self.co_num_queries))
        print("co_num_warm_qps: ", str(np.sum(self.co_num_warm_qps)))
        print("co_num_cold_qps: ", str(np.sum(self.co_num_cold_qps)))
        print("co_q_persec: ", str(np.sum(self.co_q_persec)))
        print("co_qa_cache_hits: ", str(np.sum(self.co_qa_cache_hits)))
        print("co_qa_efs_bytes_read: ", str(np.sum(self.co_qa_efs_bytes_read)))        
        print("co_qa_s3_gets: ", str(np.sum(self.co_qa_s3_gets)))                
        print("co_qa_warm_starts: ", str(np.sum(self.co_qa_warm_starts))) 
        res = json.dumps(response)
        with open(self.co_metrics_fname, 'w') as f:
            f.write(res)
            
        self.write_metrics_to_s3(response)
        
        return res
    # ----------------------------------------------------------------------------------------------------------------------------------------
    def write_metrics_to_s3(self, res):

        s3          = boto3.client('s3', region_name='eu-west-1')
        bucket      = "squash-metrics"
        subfolder   = res["dmg_params"]["fname"] + "/" + res["dmg_params"]["exp_tag"] + "/"
        filename    = res["dmg_params"]["exp_tag"] + "-" + str(res["dmg_params"]["exp_run"]) + '.csv'
        key         = subfolder + filename
        
        # print("In write_metrics_to_s3, s3 client: ", s3)
        # print("In write_metrics_to_s3, bucket: ", bucket)
        # print("In write_metrics_to_s3, key: ", key)
        
        output_list = list(res["dmg_params"].values())
        output_list.extend(list(res["co_params"].values()))
       
        csvio = StringIO()
        writer = csv.writer(csvio,quoting=csv.QUOTE_ALL)
        writer.writerow(output_list)
        # s3.put_object(Body=csvio.getvalue(), ContentType='text/csv', Bucket=bucket, Key=key)        
        
        efs_fname = os.path.join(self.dmg_params["path"], '') + res["dmg_params"]["exp_tag"] + "-" + str(res["dmg_params"]["exp_run"]) + '.csv'
        with open(efs_fname, 'w') as f:
            f.write(csvio.getvalue())
        
        csvio.close()    
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def query_allocator(self, payload):
        
        global gqa

        response = gqa.LAMBDA_CLIENT.invoke(
                    FunctionName    = gqa.QA_LAMBDA_ARN,
                    InvocationType  = 'RequestResponse',
                    Payload         = payload
                                        )
        return response
    # ----------------------------------------------------------------------------------------------------------------------------------------    
    def coordinate(self):
        
        print()
        print("Coordinator Session Begins")
        print("===========================")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Session Start Time : ", str(current_time))
        print()        

        node_gene = self.treelauncher.node_generator()        
        with ThreadPoolExecutor(max_workers=self.num_allocators) as executor:
            futures = []
            # for a_id in range(int(self.dmg_params["num_allocators"])):
            for node in node_gene:
                payload = self.build_qa_payload(node)
                futures.append( executor.submit(self.query_allocator, payload=payload) )
                                   
            for future in as_completed(futures):
                self.unload_qa_response(future.result())

        co_response = self.build_co_response()     

        self.co_end_dt = timeit.default_timer()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print()
        print("Coordinator Session End Time : ", current_time, " Elapsed : ", str(self.co_end_dt - self.co_start_dt) )
        print()
        return co_response
    # ----------------------------------------------------------------------------------------------------------------------------------------      

     
     
     
     