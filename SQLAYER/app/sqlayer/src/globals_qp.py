# --------------------------------------------------------------------------------
# Singleton Class for QueryProcessor Global Data
# --------------------------------------------------------------------------------
import boto3
import botocore

class GQP(object):

    __instance = None
    
    @staticmethod 
    def getInstance():
        """ Static access method. """
        if GQP.__instance == None:
           GQP()
        return GQP.__instance

    def __init__(self):

        """ Virtually private constructor. """
        if GQP.__instance != None:
           raise Exception("This class is a singleton!")
        else:
           GQP.__instance = self

        self.S3_CLIENT                 = boto3.client('s3', region_name='eu-west-1') 

        self.path                      = None
        self.fname                     = None
        self.at_means                  = None
        self.at_stdevs                 = None
        self.attribute_cells           = None
        self.attribute_boundary_vals   = None
        self.dim_means                 = None
        self.cov_matrix                = None
        self.transform_matrix          = None
        self.tf_dim_means              = None
        self.tf_stdevs                 = None
        self.cells                     = None
        self.boundary_vals             = None
        self.sdc_lookup_lower          = None
        self.sdc_lookup_upper          = None
        self.vaqdata                   = None
        self.bqdata                    = None
        self.quant_attr_data           = None
        self.test                      = 'Hello from g_qp_test'           

