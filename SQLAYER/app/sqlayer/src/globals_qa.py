# --------------------------------------------------------------------------------
# Singleton Class for QueryAllocator Global Data
# --------------------------------------------------------------------------------
import boto3
import botocore

class GQA(object):

    __instance = None
    
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if GQA.__instance == None:
           GQA()
        return GQA.__instance

    def __init__(self):

        """ Virtually private constructor. """
        if GQA.__instance != None:
           raise Exception("This class is a singleton!")
        else:
           GQA.__instance = self
           
        client_config = botocore.config.Config(max_pool_connections=100)

        # AWS Clients/ARNs
        self.S3_CLIENT                      = boto3.client('s3', region_name='eu-west-1') 
        self.LAMBDA_CLIENT                  = boto3.client('lambda', region_name='eu-west-1', config=client_config)
        self.QA_LAMBDA_ARN                  = 'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-allocator-lambda'
        self.QP_LAMBDA_ARNS                 = ['arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-0',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-1',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-2',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-3',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-4',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-5',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-6',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-7',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-8',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-9',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-10',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-11',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-12',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-13',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-14',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-15',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-16',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-17',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-18',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-19',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-20',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-21',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-22',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-23',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-24',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-25',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-26',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-27',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-28',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-29',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-30',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-31',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-32',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-33',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-34',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-35',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-36',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-37',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-38',
                                          'arn:aws:lambda:eu-west-1:XXXXXXXXXXXX:function:sq-processor-lambda-39']

        self.path                      = None
        self.fname                     = None
        self.partition_vectors         = None
        self.partition_ids             = None
        self.partition_pops            = None
        self.partition_centroids       = None
        self.at_means                  = None
        self.at_stdevs                 = None
        self.attribute_cells           = None
        self.attribute_boundary_vals   = None
        self.dim_means                 = None
        self.cov_matrix                = None
        self.transform_matrix          = None
        self.quant_attr_data           = None
        self.lbl_counters              = None
      #   self.lbl_vocab_sig             = None
      #   self.lbl_ds_sig                = None
        self.lbl_csrt_indices          = None
        self.lbl_csrt_indptr           = None
        
        self.test                      = 'Hello from g_qa_test'       
        
    def reset(self):
        self.path                      = None
        self.fname                     = None
        self.partition_vectors         = None
        self.partition_ids             = None
        self.partition_pops            = None
        self.partition_centroids       = None
        self.at_means                  = None
        self.at_stdevs                 = None
        self.attribute_cells           = None
        self.attribute_boundary_vals   = None
        self.dim_means                 = None
        self.cov_matrix                = None
        self.transform_matrix          = None
        self.quant_attr_data           = None
        self.lbl_counters              = None
      #   self.lbl_vocab_sig             = None
      #   self.lbl_ds_sig                = None
        self.lbl_csrt_indices          = None
        self.lbl_csrt_indptr           = None

        print("********* Resetting GQA! *********") 

