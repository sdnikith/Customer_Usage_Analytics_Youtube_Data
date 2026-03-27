"""
YouTube Analytics Pipeline DAG

This DAG orchestrates the complete YouTube analytics data pipeline including:
1. Data Ingestion (YouTube API + Kaggle)
2. PySpark ETL on AWS Glue
3. Data Quality Validation
4. NLP Video Categorization
5. ML Engagement Prediction
6. Monitoring and Alerting
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.glue import AwsGlueJobOperator
from airflow.providers.amazon.aws.operators.lambda_function import AwsLambdaInvokeFunctionOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

# Add project root to Python path
sys.path.append('/opt/airflow/dags/..')

from ingestion.youtube_api_extractor import YouTubeAPIExtractor
from ingestion.kaggle_data_loader import KaggleDataLoader
from ingestion.s3_uploader import S3Uploader
from ml.nlp_categorization import NLPVideoCategorizer
from ml.engagement_predictor import EngagementPredictor
from ml.feature_engineering import YouTubeFeatureEngineer

# Default arguments
default_args = {
    'owner': 'youtube-analytics',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Create DAG
dag = DAG(
    'youtube_analytics_pipeline',
    default_args=default_args,
    description='Complete YouTube Analytics Data Pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['youtube', 'analytics', 'ml', 'etl'],
    max_active_runs=1,
)

def extract_youtube_data(**context):
    """Extract data from YouTube API."""
    try:
        extractor = YouTubeAPIExtractor()
        
        # Extract trending videos for each region
        regions = ['US', 'GB', 'CA', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        all_data = []
        
        for region in regions:
            print(f"Extracting trending videos for region: {region}")
            trending_data = extractor.extract_trending_videos(region, max_results=50)
            if trending_data:
                all_data.extend(trending_data)
        
        # Save to local file
        import pandas as pd
        df = pd.DataFrame(all_data)
        output_path = f"/tmp/youtube_api_data_{context['ds']}.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Extracted {len(df)} records from YouTube API")
        return output_path
        
    except Exception as e:
        print(f"Error extracting YouTube data: {e}")
        raise

def load_kaggle_data(**context):
    """Load and process Kaggle YouTube dataset."""
    try:
        loader = KaggleDataLoader()
        
        # Load datasets for multiple regions
        regions = ['US', 'GB', 'CA', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        all_data = []
        
        for region in regions:
            print(f"Loading Kaggle data for region: {region}")
            region_data = loader.load_region_data(region)
            if region_data is not None:
                all_data.append(region_data)
        
        # Combine all regions
        if all_data:
            combined_data = loader.combine_regions(all_data)
            output_path = f"/tmp/kaggle_data_{context['ds']}.csv"
            combined_data.to_csv(output_path, index=False)
            
            print(f"Loaded {len(combined_data)} records from Kaggle")
            return output_path
        else:
            raise ValueError("No Kaggle data loaded")
            
    except Exception as e:
        print(f"Error loading Kaggle data: {e}")
        raise

def upload_to_s3(**context):
    """Upload extracted data to S3."""
    try:
        uploader = S3Uploader()
        
        # Get file paths from previous tasks
        youtube_file = context['task_instance'].xcom_pull(task_ids='extract_youtube_data')
        kaggle_file = context['task_instance'].xcom_pull(task_ids='load_kaggle_data')
        
        # Upload YouTube API data
        if youtube_file and os.path.exists(youtube_file):
            import pandas as pd
            df = pd.read_csv(youtube_file)
            uploader.upload_dataframe(
                df, 
                bucket=Variable.get('S3_RAW_BUCKET', 'youtube-raw'),
                key_prefix=f"youtube-api/{context['ds']}/",
                partition_cols=['region_code']
            )
            print(f"Uploaded YouTube API data to S3")
        
        # Upload Kaggle data
        if kaggle_file and os.path.exists(kaggle_file):
            import pandas as pd
            df = pd.read_csv(kaggle_file)
            uploader.upload_dataframe(
                df,
                bucket=Variable.get('S3_RAW_BUCKET', 'youtube-raw'),
                key_prefix=f"kaggle/{context['ds']}/",
                partition_cols=['region_code']
            )
            print(f"Uploaded Kaggle data to S3")
        
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

def run_nlp_categorization(**context):
    """Run NLP video categorization."""
    try:
        categorizer = NLPVideoCategorizer()
        
        # Load cleaned data from S3
        import pandas as pd
        import boto3
        
        s3_client = boto3.client('s3')
        bucket = Variable.get('S3_CLEANED_BUCKET', 'youtube-cleaned')
        
        # Get latest cleaned data file
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix='cleaned/',
            MaxKeys=1
        )
        
        if 'Contents' in response:
            latest_key = response['Contents'][0]['Key']
            
            # Download and load data
            obj = s3_client.get_object(Bucket=bucket, Key=latest_key)
            df = pd.read_parquet(obj['Body'])
            
            # Run categorization
            df_with_predictions = categorizer.predict_categories(df)
            
            # Save predictions back to S3
            output_key = latest_key.replace('cleaned/', 'nlp_predictions/')
            categorizer.save_predictions_to_s3(
                df_with_predictions, bucket, output_key
            )
            
            print(f"NLP categorization completed for {len(df)} records")
            return output_key
        else:
            raise ValueError("No cleaned data found in S3")
            
    except Exception as e:
        print(f"Error in NLP categorization: {e}")
        raise

def run_engagement_prediction(**context):
    """Run ML engagement prediction."""
    try:
        predictor = EngagementPredictor()
        
        # Load data with NLP predictions
        import pandas as pd
        import boto3
        
        s3_client = boto3.client('s3')
        bucket = Variable.get('S3_CLEANED_BUCKET', 'youtube-cleaned')
        
        # Get latest NLP predictions
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix='nlp_predictions/',
            MaxKeys=1
        )
        
        if 'Contents' in response:
            latest_key = response['Contents'][0]['Key']
            
            # Download and load data
            obj = s3_client.get_object(Bucket=bucket, Key=latest_key)
            df = pd.read_parquet(obj['Body'])
            
            # Run engagement prediction
            df_with_predictions = predictor.predict_engagement(df)
            
            # Save predictions back to S3
            output_key = latest_key.replace('nlp_predictions/', 'ml_predictions/')
            
            # Convert to parquet and upload
            buffer = df_with_predictions.to_parquet(index=False)
            s3_client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=buffer,
                ContentType='application/octet-stream'
            )
            
            print(f"Engagement prediction completed for {len(df)} records")
            return output_key
        else:
            raise ValueError("No NLP predictions found in S3")
            
    except Exception as e:
        print(f"Error in engagement prediction: {e}")
        raise

def send_success_notification(**context):
    """Send success notification to Slack."""
    return SlackWebhookOperator(
        task_id='send_success_notification',
        slack_webhook_conn_id='slack_webhook_default',
        message=f"""
✅ YouTube Analytics Pipeline Completed Successfully! 🎉

📅 Execution Date: {context['ds']}
🔧 DAG: {context['dag'].dag_id}
📊 Tasks Completed: {len(context['dag'].task_ids)}

All stages completed successfully:
- ✅ Data Ingestion (YouTube API + Kaggle)
- ✅ PySpark ETL on AWS Glue  
- ✅ Data Quality Validation
- ✅ NLP Video Categorization
- ✅ ML Engagement Prediction

📈 Data is now available in S3 for analysis and visualization!
        """,
        channel='#data-pipelines',
        username='Airflow Bot',
        icon_emoji=':chart_with_upwards_trend:'
    ).execute(context)

def send_failure_notification(context):
    """Send failure notification to Slack."""
    return SlackWebhookOperator(
        task_id='send_failure_notification',
        slack_webhook_conn_id='slack_webhook_default',
        message=f"""
❌ YouTube Analytics Pipeline Failed! 🚨

📅 Execution Date: {context['ds']}
🔧 DAG: {context['dag'].dag_id}
❌ Failed Task: {context['task_instance'].task_id}
🔍 Error: {context['task_instance'].xcom_pull(task_ids=context['task_instance'].task_id, key='return_value')}

Please investigate the failure immediately.
        """,
        channel='#data-pipelines-alerts',
        username='Airflow Bot',
        icon_emoji=':warning:'
    ).execute(context)

# Task Groups
with TaskGroup('data_ingestion', tooltip='Data Ingestion Tasks') as data_ingestion:
    extract_youtube = PythonOperator(
        task_id='extract_youtube_data',
        python_callable=extract_youtube_data,
        dag=dag,
    )
    
    load_kaggle = PythonOperator(
        task_id='load_kaggle_data',
        python_callable=load_kaggle_data,
        dag=dag,
    )
    
    upload_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
        dag=dag,
    )
    
    extract_youtube >> upload_s3
    load_kaggle >> upload_s3

# Glue ETL Task
run_glue_etl = AwsGlueJobOperator(
    task_id='run_glue_etl',
    job_name='youtube-etl-job',
    script_location=f"s3://{Variable.get('S3_SCRIPTS_BUCKET', 'youtube-scripts')}/etl/glue_etl_job.py",
    s3_bucket_logs=Variable.get('S3_LOGS_BUCKET', 'youtube-logs'),
    aws_conn_id='aws_default',
    region_name='us-west-1',
    iam_role_name='GlueServiceRole',
    create_job_kwargs={
        'GlueVersion': '4.0',
        'NumberOfWorkers': 2,
        'WorkerType': 'G.1X',
        'DefaultArguments': {
            '--JOB_NAME': 'youtube-etl-job',
            '--SOURCE_BUCKET': Variable.get('S3_RAW_BUCKET', 'youtube-raw'),
            '--TARGET_BUCKET': Variable.get('S3_CLEANED_BUCKET', 'youtube-cleaned'),
            '--DATABASE_NAME': 'youtube_analytics',
            '--TABLE_NAME': 'youtube_cleaned',
            '--TEMP_DIR': f"s3://{Variable.get('S3_TEMP_BUCKET', 'youtube-temp')}/temp/"
        }
    },
    dag=dag,
)

# Data Quality Validation
run_data_quality = AwsLambdaInvokeFunctionOperator(
    task_id='run_data_quality',
    function_name='youtube-data-quality-validator',
    aws_conn_id='aws_default',
    region_name='us-west-1',
    payload={
        'bucket': Variable.get('S3_CLEANED_BUCKET', 'youtube-cleaned'),
        'prefix': 'cleaned/',
        'execution_date': '{{ ds }}'
    },
    dag=dag,
)

# ML Tasks
with TaskGroup('ml_processing', tooltip='Machine Learning Tasks') as ml_processing:
    run_nlp = PythonOperator(
        task_id='run_nlp_categorization',
        python_callable=run_nlp_categorization,
        dag=dag,
    )
    
    run_ml = PythonOperator(
        task_id='run_engagement_prediction',
        python_callable=run_engagement_prediction,
        dag=dag,
    )
    
    run_nlp >> run_ml

# Success Notification
success_notification = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    trigger_rule='all_success',
    dag=dag,
)

# Define task dependencies
data_ingestion >> run_glue_etl
run_glue_etl >> run_data_quality
run_data_quality >> ml_processing
ml_processing >> success_notification

# Set up failure notification
for task in dag.tasks:
    task.on_failure_callback = send_failure_notification
