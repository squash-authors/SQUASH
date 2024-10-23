import os
import pathlib
from glob import glob
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import progressbar
import argparse

BASE_DIR = "/XX/XX/squash_datasets/reqfile_zips/"
AWS_REGION = "eu-west-1"
SQUASH_BUCKET = "squash-data-bucket"
S3_PREFIX = "reqfile_zips/"

# s3 = boto3.client("s3", region_name=AWS_REGION)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for SQUASH Boto3 S3 Upload")
    
    parser.add_argument(
        "--dataset",
        default = 'gist1m'
    )

    parser.add_argument(
        "--zip_name",
        default = 'gist1m_bb3840_p10_csuint8_30iter.zip'
    )

    return parser.parse_args()


def upload_to_aws(local_file, s3_bucket, s3_folder, s3_filename, force_overwrite='n'):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    print("In upload_to_aws!")
    print("local_file: ",   local_file)
    print("s3_bucket: ",    s3_bucket)
    print("s3_folder: ",    s3_folder)
    print("s3_filename: ",  s3_filename)

    def write_to_aws():
        print("In write_to_aws!")
        statinfo = os.stat(local_file)
        up_progress = progressbar.progressbar.ProgressBar(maxval=statinfo.st_size)
        up_progress.start()

        def upload_progress(chunk):
            up_progress.update(up_progress.currval + chunk)


        try:
            print("Writing")
            print("s3 key: ", str(s3_folder + s3_filename))
            s3.upload_file(local_file, s3_bucket, s3_folder+s3_filename, Callback=upload_progress)
            print()
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The source file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False
    try:
        s3.head_object(Bucket=s3_bucket, Key=s3_folder+s3_filename)
        if force_overwrite=='y':
            write_to_aws
        else:
            ask_overwrite = input('File already exists at destination. Overwrite?')
            if ask_overwrite == 'y':
                write_to_aws()
            else:
                print('leaving application')
    except ClientError as e:
        write_to_aws()



if __name__ == '__main__':

    args = parse_args()
    local_zip_file = BASE_DIR + args.dataset + "/" + args.zip_name # "/XX/X/squash_datasets/reqfile_zips/" + "sift1m/" + "sift1m....zip"
    s3_dir = S3_PREFIX + args.dataset + "/" # "reqfile_zips/" + "sift1m/"

    upload_to_aws(local_zip_file, SQUASH_BUCKET, s3_dir, args.zip_name)










