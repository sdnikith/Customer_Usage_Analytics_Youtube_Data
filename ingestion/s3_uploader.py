"""
S3 Uploader with Lifecycle Policies

This module handles uploading YouTube data to S3 with proper partitioning,
lifecycle policies, error handling, and retry logic.
"""

import os
import json
import logging
import boto3
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
from botocore.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Uploader:
    """
    Uploads YouTube data to S3 with partitioning and lifecycle management.
    
    Attributes:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
    """
    
    def __init__(self, bucket_name: Optional[str] = None, region: str = "us-west-1"):
        """
        Initialize S3 uploader.
        
        Args:
            bucket_name (Optional[str]): S3 bucket name. If None, loads from environment.
            region (str): AWS region. Defaults to "us-west-1".
            
        Raises:
            ValueError: If bucket name is not provided or found in environment.
            NoCredentialsError: If AWS credentials are not found.
        """
        self.bucket_name = bucket_name or os.getenv('S3_RAW_BUCKET')
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided or set in S3_RAW_BUCKET environment variable")
        
        self.max_retries = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY_SECONDS', '5'))
        
        # Configure boto3 with retry settings
        config = Config(
            region_name=region,
            retries={
                'max_attempts': self.max_retries,
                'mode': 'adaptive'
            }
        )
        
        try:
            self.s3_client = boto3.client('s3', config=config)
            self.s3_resource = boto3.resource('s3', config=config)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"Bucket {self.bucket_name} does not exist")
                raise
            else:
                logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")
            raise
    
    def create_bucket_if_not_exists(self) -> bool:
        """
        Create S3 bucket if it doesn't exist.
        
        Returns:
            bool: True if bucket exists or was created successfully
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    if self.s3_client.meta.region_name == 'us-east-1':
                        # us-east-1 doesn't require LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.s3_client.meta.region_name
                            }
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                    return True
                except Exception as create_error:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_error}")
                    return False
            else:
                logger.error(f"Error checking bucket existence: {e}")
                return False
    
    def setup_lifecycle_policy(self) -> bool:
        """
        Setup lifecycle policy for the bucket.
        
        Policy:
        - Standard → Infrequent Access after 30 days
        - Infrequent Access → Glacier after 90 days
        
        Returns:
            bool: True if policy was set successfully
        """
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'YouTubeDataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {
                        'Prefix': ''
                    },
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            logger.info("Lifecycle policy configured successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set lifecycle policy: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, region: str, date_str: Optional[str] = None,
                        file_format: str = 'parquet') -> bool:
        """
        Upload DataFrame to S3 with partitioning.
        
        Args:
            df (pd.DataFrame): DataFrame to upload
            region (str): Region code for partitioning
            date_str (Optional[str]): Date string for partitioning. If None, uses current date.
            file_format (str): File format ('parquet', 'csv', 'json')
            
        Returns:
            bool: True if upload was successful
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Create partitioned path
        prefix = f"{region}/{date_str}/"
        
        try:
            if file_format.lower() == 'parquet':
                return self._upload_parquet(df, prefix)
            elif file_format.lower() == 'csv':
                return self._upload_csv(df, prefix)
            elif file_format.lower() == 'json':
                return self._upload_json(df, prefix)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading DataFrame: {e}")
            return False
    
    def _upload_parquet(self, df: pd.DataFrame, prefix: str) -> bool:
        """
        Upload DataFrame as Parquet file.
        
        Args:
            df (pd.DataFrame): DataFrame to upload
            prefix (str): S3 prefix
            
        Returns:
            bool: True if upload was successful
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert DataFrame to Arrow Table
            table = pa.Table.from_pandas(df)
            
            # Create buffer
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression='snappy')
            
            # Upload to S3
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            key = f"{prefix}youtube_data_{timestamp}.parquet"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue().to_pybytes(),
                ContentType='application/octet-stream',
                Metadata={
                    'record_count': str(len(df)),
                    'upload_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Uploaded {len(df)} records to s3://{self.bucket_name}/{key}")
            return True
            
        except ImportError:
            logger.error("PyArrow not installed. Install with: pip install pyarrow")
            return False
        except Exception as e:
            logger.error(f"Error uploading Parquet file: {e}")
            return False
    
    def _upload_csv(self, df: pd.DataFrame, prefix: str) -> bool:
        """
        Upload DataFrame as CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to upload
            prefix (str): S3 prefix
            
        Returns:
            bool: True if upload was successful
        """
        try:
            # Convert to CSV
            csv_buffer = df.to_csv(index=False)
            
            # Upload to S3
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            key = f"{prefix}youtube_data_{timestamp}.csv"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=csv_buffer.encode('utf-8'),
                ContentType='text/csv',
                Metadata={
                    'record_count': str(len(df)),
                    'upload_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Uploaded {len(df)} records to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading CSV file: {e}")
            return False
    
    def _upload_json(self, df: pd.DataFrame, prefix: str) -> bool:
        """
        Upload DataFrame as JSON file.
        
        Args:
            df (pd.DataFrame): DataFrame to upload
            prefix (str): S3 prefix
            
        Returns:
            bool: True if upload was successful
        """
        try:
            # Convert to JSON
            json_buffer = df.to_json(orient='records', date_format='iso', indent=2)
            
            # Upload to S3
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            key = f"{prefix}youtube_data_{timestamp}.json"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_buffer.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'record_count': str(len(df)),
                    'upload_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Uploaded {len(df)} records to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading JSON file: {e}")
            return False
    
    def upload_json_data(self, data: Union[Dict, List], region: str, date_str: Optional[str] = None,
                        filename: Optional[str] = None) -> bool:
        """
        Upload JSON data to S3.
        
        Args:
            data (Union[Dict, List]): JSON data to upload
            region (str): Region code for partitioning
            date_str (Optional[str]): Date string for partitioning
            filename (Optional[str]): Custom filename
            
        Returns:
            bool: True if upload was successful
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"youtube_data_{timestamp}.json"
        
        # Create partitioned path
        key = f"{region}/{date_str}/{filename}"
        
        try:
            json_string = json.dumps(data, indent=2, default=str)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_string.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'upload_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Uploaded JSON data to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading JSON data: {e}")
            return False
    
    def list_uploaded_files(self, region: Optional[str] = None, 
                           date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List uploaded files in the bucket.
        
        Args:
            region (Optional[str]): Filter by region
            date_str (Optional[str]): Filter by date
            
        Returns:
            List[Dict[str, Any]]: List of file information
        """
        try:
            # Build prefix
            prefix = ""
            if region:
                prefix += f"{region}/"
            if date_str:
                prefix += f"{date_str}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                file_info = {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                }
                files.append(file_info)
            
            logger.info(f"Found {len(files)} files with prefix '{prefix}'")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def get_bucket_info(self) -> Dict[str, Any]:
        """
        Get bucket information and statistics.
        
        Returns:
            Dict[str, Any]: Bucket information
        """
        try:
            # Get bucket location
            location = self.s3_client.get_bucket_location(Bucket=self.bucket_name)
            
            # Get bucket size and object count
            total_size = 0
            object_count = 0
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                for obj in page.get('Contents', []):
                    total_size += obj['Size']
                    object_count += 1
            
            # Get lifecycle configuration
            try:
                lifecycle = self.s3_client.get_bucket_lifecycle_configuration(Bucket=self.bucket_name)
                has_lifecycle = True
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
                    has_lifecycle = False
                else:
                    raise
            
            info = {
                'bucket_name': self.bucket_name,
                'region': location.get('LocationConstraint') or 'us-east-1',
                'total_objects': object_count,
                'total_size_bytes': total_size,
                'total_size_gb': round(total_size / (1024**3), 2),
                'has_lifecycle_policy': has_lifecycle,
                'checked_at': datetime.now(timezone.utc).isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting bucket info: {e}")
            return {}
    
    def delete_file(self, key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            key (str): S3 object key
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted file: s3://{self.bucket_name}/{key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {key}: {e}")
            return False


def main():
    """
    Main function to test the S3 uploader.
    """
    try:
        import pandas as pd
        
        # Create sample data
        sample_data = pd.DataFrame({
            'video_id': ['test1', 'test2', 'test3'],
            'title': ['Test Video 1', 'Test Video 2', 'Test Video 3'],
            'views': [1000, 2000, 3000],
            'likes': [100, 200, 300],
            'comments': [10, 20, 30]
        })
        
        # Initialize uploader
        uploader = S3Uploader()
        
        # Create bucket if needed
        uploader.create_bucket_if_not_exists()
        
        # Setup lifecycle policy
        uploader.setup_lifecycle_policy()
        
        # Upload sample data
        success = uploader.upload_dataframe(sample_data, region='US', file_format='parquet')
        if success:
            print("Sample data uploaded successfully")
        
        # Get bucket info
        info = uploader.get_bucket_info()
        print("Bucket Info:")
        print(json.dumps(info, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
