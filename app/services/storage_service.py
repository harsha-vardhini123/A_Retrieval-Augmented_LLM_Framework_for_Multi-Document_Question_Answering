import boto3
from botocore.exceptions import ClientError
from config import config

class S3Storage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_SECRET_KEY,
            region_name='us-east-1'
        )
        self.bucket = config.AWS_BUCKET_KEY

    def upload_file(self, file_obj, filename):
        try:
            self.s3.upload_fileobj(file_obj, self.bucket, filename)
            return True
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False

    def upload_bytes(self, byte_stream, key):
        try:
            self.s3.upload_fileobj(byte_stream, self.bucket, key)
            return True
        except ClientError as e:
            print(f"Error uploading bytes: {e}")
            return False

    def get_object(self, key):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response['Body']
        except ClientError as e:
            print(f"Error retrieving file: {e}")
            return None
