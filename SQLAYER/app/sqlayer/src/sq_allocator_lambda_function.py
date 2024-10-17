import json
import boto3
import timeit
from datetime import datetime
import sys
sys.path.append('src')

from sqlayer import QueryAllocator as QueryAllocator
from sqlayer import GQA
global gqa
gqa = GQA.getInstance()

# ----------------------------------------------------------------------------------------------------------------------------------------
def lambda_handler(event, context):
    
    lambda_handler_start_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("QA Start Time : ", str(current_time))    
    print()

    queryallocator = QueryAllocator(payload=event)
    qa_response = queryallocator.allocate()

    lambda_handler_end_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("QA Session End Time : ", current_time, " Elapsed : ", str(lambda_handler_end_dt - lambda_handler_start_dt) )
    print()

    # return ({
    #     "statusCode": 200,
    #     "body": qa_response
    #     })
    
    return qa_response