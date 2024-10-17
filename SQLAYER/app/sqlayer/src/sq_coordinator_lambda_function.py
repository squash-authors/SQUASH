import json
import boto3
import timeit
from datetime import datetime
import sys
sys.path.append('src')

from sqlayer import Coordinator as Coordinator
from sqlayer import QueryAllocator as QueryAllocator
from sqlayer import GQA
global gqa
gqa = GQA.getInstance()

# ----------------------------------------------------------------------------------------------------------------------------------------
def lambda_handler(event, context):
    
    lambda_handler_start_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("SQUASH Session Start Time : ", str(current_time))    
    print()

    mode            = event["dmg_params"]["mode"]
    num_allocators  = int(event["dmg_params"]["num_allocators"])    
    
    # Instantiate and run QueryAllocator or Coordinator
    if mode == 'P': 
        queryallocator = QueryAllocator(payload=event)
        queryallocator.prvd_processor()
    elif num_allocators == 1:
        queryallocator = QueryAllocator(payload=event)
        response = queryallocator.allocate()
    else:
        coordinator = Coordinator(payload=event)        
        response = coordinator.coordinate()    

    lambda_handler_end_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("SQUASH Session End Time : ", current_time, " Elapsed : ", str(lambda_handler_end_dt - lambda_handler_start_dt) )
    print()

    return {
            "statusCode": 200
           }
    