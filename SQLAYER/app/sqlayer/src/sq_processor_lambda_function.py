import json
import boto3
import timeit
from datetime import datetime
import sys
sys.path.append('src')

from sqlayer import QueryProcessor as QueryProcessor
from sqlayer import GQP
global gqp
gqp = GQP.getInstance()

# ----------------------------------------------------------------------------------------------------------------------------------------
def lambda_handler(event, context):
    
    lambda_handler_start_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("QP Start Time : ", str(current_time))    
    print()

    queryprocessor = QueryProcessor(payload=event)
    qp_response = queryprocessor.process()

    lambda_handler_end_dt = timeit.default_timer()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print()
    print("QP Session End Time : ", current_time, " Elapsed : ", str(lambda_handler_end_dt - lambda_handler_start_dt) )
    print()

    # return ({
    #     "statusCode": 200,
    #     "body": qp_response
    #     })
      
    return qp_response