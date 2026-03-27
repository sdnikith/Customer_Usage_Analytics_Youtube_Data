"""
AWS Lambda Data Quality Validator

This Lambda function is triggered by S3 PutObject events on the cleaned bucket,
runs Great Expectations validation, saves reports, publishes CloudWatch metrics,
and sends SNS alerts if any expectations fail.
"""

import json
import os
import logging
import boto3
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from urllib.parse import unquote_plus
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import BaseDataContext

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
S3_QUALITY_REPORTS_BUCKET = os.getenv('S3_QUALITY_REPORTS_BUCKET', 'youtube-quality-reports')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN')
DATA_QUALITY_THRESHOLD = float(os.getenv('DATA_QUALITY_THRESHOLD', '90'))
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')


class DataQualityValidator:
    """
    Data quality validator using Great Expectations.
    
    Attributes:
        s3_client: S3 client
        sns_client: SNS client
        cloudwatch_client: CloudWatch client
        data_context: Great Expectations data context
    """
    
    def __init__(self):
        """Initialize the validator with AWS clients and GE context."""
        self.s3_client = boto3.client('s3')
        self.sns_client = boto3.client('sns')
        self.cloudwatch_client = boto3.client('cloudwatch')
        
        # Initialize Great Expectations context
        try:
            self.data_context = BaseDataContext()
            logger.info("Great Expectations context initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Great Expectations context: {e}")
            raise
    
    def parse_s3_event(self, event: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Parse S3 event records to extract bucket and key information.
        
        Args:
            event (Dict[str, Any]): S3 event
            
        Returns:
            List[Dict[str, str]]: List of bucket and key information
        """
        s3_records = []
        
        for record in event.get('Records', []):
            if record.get('eventSource') == 'aws:s3':
                s3_info = record.get('s3', {})
                bucket = s3_info.get('bucket', {}).get('name')
                key = unquote_plus(s3_info.get('object', {}).get('key', ''))
                
                if bucket and key:
                    s3_records.append({'bucket': bucket, 'key': key})
                    logger.info(f"Found S3 record: bucket={bucket}, key={key}")
        
        return s3_records
    
    def load_data_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """
        Load data from S3 into pandas DataFrame.
        
        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            Exception: If data loading fails
        """
        try:
            # Determine file format from key
            if key.endswith('.parquet'):
                # Use pandas to read parquet from S3
                import io
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                buffer = io.BytesIO(response['Body'].read())
                df = pd.read_parquet(buffer)
                
            elif key.endswith('.csv'):
                # Use pandas to read CSV from S3
                import io
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                buffer = io.StringIO(response['Body'].read().decode('utf-8'))
                df = pd.read_csv(buffer)
                
            elif key.endswith('.json'):
                # Use pandas to read JSON from S3
                import io
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                buffer = io.StringIO(response['Body'].read().decode('utf-8'))
                df = pd.read_json(buffer)
                
            else:
                raise ValueError(f"Unsupported file format: {key}")
            
            logger.info(f"Loaded {len(df)} records from s3://{bucket}/{key}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from s3://{bucket}/{key}: {e}")
            raise
    
    def run_great_expectations_validation(self, df: pd.DataFrame, file_key: str) -> Dict[str, Any]:
        """
        Run Great Expectations validation on the data.
        
        Args:
            df (pd.DataFrame): Data to validate
            file_key (str): Original file key for reference
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Create Great Expectations dataset
            ge_df = ge.from_pandas(df)
            
            # Load expectation suite
            suite_name = "youtube_data_suite"
            try:
                expectation_suite = self.data_context.get_expectation_suite(suite_name)
                if not expectation_suite:
                    logger.warning(f"Expectation suite {suite_name} not found, using default validations")
                    expectation_suite = self._create_default_suite()
            except Exception as e:
                logger.warning(f"Error loading expectation suite: {e}, using default validations")
                expectation_suite = self._create_default_suite()
            
            # Run validation
            validation_result = ge_df.validate(expectation_suite=expectation_suite)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result)
            
            # Prepare results
            results = {
                'validation_result': validation_result,
                'quality_score': quality_score,
                'file_key': file_key,
                'record_count': len(df),
                'validation_timestamp': datetime.now(timezone.utc).isoformat(),
                'success': quality_score >= DATA_QUALITY_THRESHOLD
            }
            
            logger.info(f"Validation completed. Quality score: {quality_score:.2f}%, Success: {results['success']}")
            return results
            
        except Exception as e:
            logger.error(f"Error running Great Expectations validation: {e}")
            raise
    
    def _create_default_suite(self):
        """Create a default expectation suite if none exists."""
        # This is a fallback - in production, the suite should be properly configured
        logger.warning("Creating default expectation suite - this should be pre-configured")
        return None
    
    def _calculate_quality_score(self, validation_result) -> float:
        """
        Calculate overall data quality score from validation results.
        
        Args:
            validation_result: Great Expectations validation result
            
        Returns:
            float: Quality score (0-100)
        """
        if not validation_result or not hasattr(validation_result, 'results'):
            return 0.0
        
        total_expectations = len(validation_result.results)
        if total_expectations == 0:
            return 0.0
        
        successful_expectations = sum(
            1 for result in validation_result.results 
            if result.success
        )
        
        quality_score = (successful_expectations / total_expectations) * 100
        return round(quality_score, 2)
    
    def save_validation_report(self, results: Dict[str, Any]) -> None:
        """
        Save validation report to S3.
        
        Args:
            results (Dict[str, Any]): Validation results
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            
            # Save JSON report
            json_report = {
                'quality_score': results['quality_score'],
                'success': results['success'],
                'record_count': results['record_count'],
                'file_key': results['file_key'],
                'validation_timestamp': results['validation_timestamp'],
                'validation_results': str(results['validation_result'])
            }
            
            json_key = f"quality-reports/{timestamp}_validation_report.json"
            self.s3_client.put_object(
                Bucket=S3_QUALITY_REPORTS_BUCKET,
                Key=json_key,
                Body=json.dumps(json_report, indent=2, default=str),
                ContentType='application/json'
            )
            
            # Save HTML report (simplified version)
            html_report = self._generate_html_report(results)
            html_key = f"quality-reports/{timestamp}_validation_report.html"
            self.s3_client.put_object(
                Bucket=S3_QUALITY_REPORTS_BUCKET,
                Key=html_key,
                Body=html_report,
                ContentType='text/html'
            )
            
            logger.info(f"Validation reports saved to s3://{S3_QUALITY_REPORTS_BUCKET}/quality-reports/")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
            # Don't raise - this is non-critical
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML validation report.
        
        Args:
            results (Dict[str, Any]): Validation results
            
        Returns:
            str: HTML report
        """
        quality_score = results['quality_score']
        success = results['success']
        
        # Determine status color
        if success:
            status_color = "green"
            status_text = "PASS"
        else:
            status_color = "red"
            status_text = "FAIL"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YouTube Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .status {{ color: {status_color}; font-size: 24px; font-weight: bold; }}
                .metric {{ margin: 10px 0; }}
                .details {{ margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YouTube Data Quality Validation Report</h1>
                <div class="status">Status: {status_text}</div>
                <div class="metric">Quality Score: {quality_score}%</div>
                <div class="metric">Record Count: {results['record_count']}</div>
                <div class="metric">File: {results['file_key']}</div>
                <div class="metric">Timestamp: {results['validation_timestamp']}</div>
            </div>
            
            <div class="details">
                <h2>Validation Details</h2>
                <p>Full validation results are available in the JSON report.</p>
                <p>This report was generated automatically by the YouTube Analytics data quality pipeline.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def publish_cloudwatch_metric(self, quality_score: float, success: bool) -> None:
        """
        Publish data quality metrics to CloudWatch.
        
        Args:
            quality_score (float): Data quality score
            success (bool): Whether validation passed
        """
        try:
            # Publish quality score metric
            self.cloudwatch_client.put_metric_data(
                Namespace='YouTube/Analytics',
                MetricData=[
                    {
                        'MetricName': 'DataQualityScore',
                        'Value': quality_score,
                        'Unit': 'Percent',
                        'Timestamp': datetime.now(timezone.utc),
                        'Dimensions': [
                            {
                                'Name': 'DataSource',
                                'Value': 'YouTubeCleaned'
                            }
                        ]
                    }
                ]
            )
            
            # Publish validation success metric
            self.cloudwatch_client.put_metric_data(
                Namespace='YouTube/Analytics',
                MetricData=[
                    {
                        'MetricName': 'ValidationSuccess',
                        'Value': 1 if success else 0,
                        'Unit': 'Count',
                        'Timestamp': datetime.now(timezone.utc),
                        'Dimensions': [
                            {
                                'Name': 'DataSource',
                                'Value': 'YouTubeCleaned'
                            }
                        ]
                    }
                ]
            )
            
            logger.info(f"Published CloudWatch metrics: QualityScore={quality_score}, Success={success}")
            
        except Exception as e:
            logger.error(f"Error publishing CloudWatch metrics: {e}")
            # Don't raise - this is non-critical
    
    def send_sns_alert(self, results: Dict[str, Any]) -> None:
        """
        Send SNS alert if validation fails.
        
        Args:
            results (Dict[str, Any]): Validation results
        """
        if not SNS_TOPIC_ARN:
            logger.warning("SNS_TOPIC_ARN not configured, skipping alert")
            return
        
        # Only send alerts for failures
        if results['success']:
            logger.info("Validation passed, not sending alert")
            return
        
        try:
            message = f"""
YouTube Data Quality Validation FAILED

File: {results['file_key']}
Quality Score: {results['quality_score']}%
Record Count: {results['record_count']}
Timestamp: {results['validation_timestamp']}

Please investigate the data quality issues immediately.

This alert was generated by the YouTube Analytics pipeline.
            """
            
            subject = f"ALERT: YouTube Data Quality Validation Failed - {results['quality_score']}%"
            
            self.sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message.strip(),
                Subject=subject
            )
            
            logger.info("SNS alert sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending SNS alert: {e}")
            # Don't raise - this is non-critical
    
    def send_slack_notification(self, results: Dict[str, Any]) -> None:
        """
        Send Slack notification (if configured).
        
        Args:
            results (Dict[str, Any]): Validation results
        """
        if not SLACK_WEBHOOK_URL:
            logger.info("Slack webhook not configured, skipping notification")
            return
        
        try:
            import requests
            
            color = "good" if results['success'] else "danger"
            status = "✅ PASSED" if results['success'] else "❌ FAILED"
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": "YouTube Data Quality Validation",
                        "fields": [
                            {
                                "title": "Status",
                                "value": status,
                                "short": True
                            },
                            {
                                "title": "Quality Score",
                                "value": f"{results['quality_score']}%",
                                "short": True
                            },
                            {
                                "title": "Record Count",
                                "value": f"{results['record_count']:,}",
                                "short": True
                            },
                            {
                                "title": "File",
                                "value": results['file_key'],
                                "short": True
                            }
                        ],
                        "footer": "YouTube Analytics Pipeline",
                        "ts": int(datetime.now(timezone.utc).timestamp())
                    }
                ]
            }
            
            response = requests.post(SLACK_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            # Don't raise - this is non-critical
    
    def validate_file(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Validate a single file.
        
        Args:
            bucket (str): S3 bucket
            key (str): S3 key
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info(f"Starting validation for s3://{bucket}/{key}")
        
        # Load data
        df = self.load_data_from_s3(bucket, key)
        
        # Run validation
        results = self.run_great_expectations_validation(df, key)
        
        # Save report
        self.save_validation_report(results)
        
        # Publish metrics
        self.publish_cloudwatch_metric(results['quality_score'], results['success'])
        
        # Send alerts if needed
        if not results['success']:
            self.send_sns_alert(results)
        
        # Send Slack notification
        self.send_slack_notification(results)
        
        return results


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event (Dict[str, Any]): Lambda event (S3 PutObject)
        context (Any): Lambda context
        
    Returns:
        Dict[str, Any]: Response
    """
    try:
        logger.info("Data quality validator Lambda function started")
        
        # Initialize validator
        validator = DataQualityValidator()
        
        # Parse S3 event
        s3_records = validator.parse_s3_event(event)
        
        if not s3_records:
            logger.warning("No S3 records found in event")
            return {
                'statusCode': 200,
                'body': 'No S3 records to process'
            }
        
        # Process each file
        validation_results = []
        for record in s3_records:
            try:
                result = validator.validate_file(record['bucket'], record['key'])
                validation_results.append(result)
                
            except Exception as e:
                logger.error(f"Error validating file {record['key']}: {e}")
                validation_results.append({
                    'file_key': record['key'],
                    'error': str(e),
                    'success': False
                })
        
        # Prepare response
        successful_validations = sum(1 for r in validation_results if r.get('success', False))
        total_validations = len(validation_results)
        
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {total_validations} files, {successful_validations} successful',
                'results': validation_results
            })
        }
        
        logger.info(f"Lambda function completed successfully: {response['body']}")
        return response
        
    except Exception as e:
        logger.error(f"Unhandled error in Lambda function: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error during data quality validation'
            })
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "Records": [
            {
                "eventSource": "aws:s3",
                "s3": {
                    "bucket": {"name": "youtube-cleaned"},
                    "object": {"key": "test_data.parquet"}
                }
            }
        ]
    }
    
    # Mock context
    class MockContext:
        def __init__(self):
            self.aws_request_id = "test-request-id"
            self.function_name = "data-quality-validator"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-west-1:123456789012:function:data-quality-validator"
            self.memory_limit_in_mb = 512
            self.get_remaining_time_in_millis = lambda: 30000
    
    # Run handler
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
