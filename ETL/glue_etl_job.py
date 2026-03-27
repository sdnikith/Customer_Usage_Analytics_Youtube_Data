"""
AWS Glue ETL Job for YouTube Data Processing

This PySpark job processes 100,000+ video records, cleans data, performs transformations,
converts to Parquet format, and registers in Glue Data Catalog.
"""

import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, lit, when, count, sum as spark_sum, avg, 
    datediff, to_date, regexp_replace, split, length,
    hour, dayofweek, weekofyear, month, year,
    udf, pandas_udf, PandasUDFType, size
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    LongType, DoubleType, TimestampType, BooleanType,
    ArrayType
)
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeGlueETL:
    """
    AWS Glue ETL job for processing YouTube data.
    
    Attributes:
        glue_context: Glue context object
        spark_session: Spark session object
        job: Glue job object
        args: Job arguments
    """
    
    def __init__(self):
        """Initialize the Glue ETL job."""
        try:
            # Get job arguments
            args = getResolvedOptions(sys.argv, [
                'JOB_NAME',
                'SOURCE_BUCKET',
                'TARGET_BUCKET',
                'DATABASE_NAME',
                'TABLE_NAME',
                'TEMP_DIR'
            ])
            
            # Initialize Spark and Glue contexts
            sc = SparkContext()
            glue_context = GlueContext(sc)
            spark_session = glue_context.spark_session
            
            # Initialize job
            job = Job(glue_context)
            job.init(args['JOB_NAME'], args)
            
            self.glue_context = glue_context
            self.spark_session = spark_session
            self.job = job
            self.args = args
            
            logger.info(f"Glue ETL job initialized: {args['JOB_NAME']}")
            
        except Exception as e:
            logger.error(f"Error initializing Glue ETL job: {e}")
            raise
    
    def define_schema(self) -> StructType:
        """
        Define the expected schema for YouTube data.
        
        Returns:
            StructType: Spark schema definition
        """
        schema = StructType([
            StructField("video_id", StringType(), nullable=False),
            StructField("title", StringType(), nullable=False),
            StructField("description", StringType(), nullable=True),
            StructField("tags", StringType(), nullable=True),
            StructField("tag_count", IntegerType(), nullable=True),
            StructField("category_id", IntegerType(), nullable=False),
            StructField("category_name", StringType(), nullable=True),
            StructField("channel_id", StringType(), nullable=True),
            StructField("channel_title", StringType(), nullable=False),
            StructField("publish_date", TimestampType(), nullable=True),
            StructField("trending_date", TimestampType(), nullable=False),
            StructField("region_code", StringType(), nullable=False),
            StructField("views", LongType(), nullable=False),
            StructField("likes", LongType(), nullable=False),
            StructField("comments", LongType(), nullable=False),
            StructField("engagement_rate", DoubleType(), nullable=True),
            StructField("title_length", IntegerType(), nullable=True),
            StructField("description_length", IntegerType(), nullable=True),
            StructField("title_word_count", IntegerType(), nullable=True),
            StructField("thumbnail_url", StringType(), nullable=True),
            StructField("extracted_at", TimestampType(), nullable=True)
        ])
        return schema
    
    def load_data_from_s3(self, source_path: str) -> DataFrame:
        """
        Load data from S3 in various formats.
        
        Args:
            source_path (str): S3 path to source data
            
        Returns:
            DataFrame: Loaded data
        """
        logger.info(f"Loading data from S3: {source_path}")
        
        try:
            # Try different formats
            formats = ['parquet', 'json', 'csv']
            df = None
            
            for fmt in formats:
                try:
                    if fmt == 'parquet':
                        df = self.spark_session.read.parquet(source_path)
                    elif fmt == 'json':
                        df = self.spark_session.read.json(source_path)
                    elif fmt == 'csv':
                        df = self.spark_session.read.option('header', 'true').csv(source_path)
                    
                    logger.info(f"Successfully loaded data as {fmt}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load as {fmt}: {e}")
                    continue
            
            if df is None:
                raise ValueError("Could not load data in any supported format")
            
            # Apply schema if needed
            expected_schema = self.define_schema()
            df = df.select([field.name for field in expected_schema.fields])
            
            logger.info(f"Loaded {df.count()} records with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from S3: {e}")
            raise
    
    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Clean the input data by removing nulls, duplicates, and standardizing formats.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        
        initial_count = df.count()
        logger.info(f"Initial record count: {initial_count}")
        
        # Remove records with null critical fields
        critical_fields = ['video_id', 'title', 'channel_title', 'views', 'likes']
        df_clean = df.na.drop(subset=critical_fields)
        
        after_null_removal = df_clean.count()
        logger.info(f"After null removal: {after_null_removal} (removed {initial_count - after_null_removal})")
        
        # Remove duplicates based on video_id
        df_clean = df_clean.dropDuplicates(['video_id'])
        
        after_dedup = df_clean.count()
        logger.info(f"After deduplication: {after_dedup} (removed {after_null_removal - after_dedup})")
        
        # Standardize date formats
        df_clean = df_clean.withColumn(
            'publish_date',
            when(col('publish_date').isNull(), lit(None))
            .otherwise(col('publish_date'))
        )
        
        # Handle missing descriptions
        df_clean = df_clean.withColumn(
            'description',
            when(col('description').isNull() | (col('description') == ''), 'No description available')
            .otherwise(col('description'))
        )
        
        # Handle missing tags
        df_clean = df_clean.withColumn(
            'tags',
            when(col('tags').isNull() | (col('tags') == ''), '')
            .otherwise(col('tags'))
        )
        
        # Clean text fields
        df_clean = df_clean.withColumn('title', regexp_replace(col('title'), r'\s+', ' '))
        df_clean = df_clean.withColumn('description', regexp_replace(col('description'), r'\s+', ' '))
        
        # Ensure numeric fields are properly typed
        numeric_fields = ['views', 'likes', 'comments', 'tag_count', 'title_length', 
                         'description_length', 'title_word_count']
        
        for field in numeric_fields:
            if field in df_clean.columns:
                df_clean = df_clean.withColumn(
                    field,
                    when(col(field).isNull(), 0)
                    .otherwise(col(field).cast('long'))
                )
        
        # Ensure category_id is valid (1-44)
        df_clean = df_clean.withColumn(
            'category_id',
            when((col('category_id') < 1) | (col('category_id') > 44), lit(1))
            .otherwise(col('category_id'))
        )
        
        final_count = df_clean.count()
        logger.info(f"Final cleaned record count: {final_count}")
        
        return df_clean
    
    def transform_data(self, df: DataFrame) -> DataFrame:
        """
        Transform data by adding calculated fields and features.
        
        Args:
            df (DataFrame): Cleaned DataFrame
            
        Returns:
            DataFrame: Transformed DataFrame
        """
        logger.info("Starting data transformation")
        
        # Calculate engagement rate if not present
        if 'engagement_rate' not in df.columns:
            df = df.withColumn(
                'engagement_rate',
                when(col('views') > 0, (col('likes') + col('comments')).cast('double') / col('views').cast('double'))
                .otherwise(lit(0.0))
            )
        
        # Calculate time to trending (in hours)
        df = df.withColumn(
            'time_to_trending_hours',
            when(
                col('publish_date').isNotNull() & col('trending_date').isNotNull(),
                (datediff(col('trending_date'), col('publish_date')) * 24.0)
            ).otherwise(lit(None))
        )
        
        # Extract temporal features
        df = df.withColumn('publish_hour', hour(col('publish_date')))
        df = df.withColumn('publish_day_of_week', dayofweek(col('publish_date')))
        df = df.withColumn('publish_week', weekofyear(col('publish_date')))
        df = df.withColumn('publish_month', month(col('publish_date')))
        df = df.withColumn('publish_year', year(col('publish_date')))
        
        # Calculate title and description metrics if not present
        if 'title_word_count' not in df.columns:
            df = df.withColumn('title_word_count', size(split(col('title'), ' ')))
        
        if 'title_length' not in df.columns:
            df = df.withColumn('title_length', length(col('title')))
        
        if 'description_length' not in df.columns:
            df = df.withColumn('description_length', length(col('description')))
        
        if 'tag_count' not in df.columns:
            df = df.withColumn(
                'tag_count',
                when(col('tags').isNull() | (col('tags') == ''), lit(0))
                .otherwise(size(split(col('tags'), '[|,]')))
            )
        
        # Add boolean flags
        df = df.withColumn(
            'has_description',
            when(col('description').isNotNull() & (col('description') != 'No description available'), lit(True))
            .otherwise(lit(False))
        )
        
        df = df.withColumn(
            'has_tags',
            when(col('tag_count') > 0, lit(True))
            .otherwise(lit(False))
        )
        
        # Add engagement categories
        df = df.withColumn(
            'engagement_category',
            when(col('engagement_rate') < 0.01, 'Low')
            .when(col('engagement_rate') < 0.05, 'Medium')
            .otherwise('High')
        )
        
        # Add view categories
        df = df.withColumn(
            'view_category',
            when(col('views') < 10000, 'Low')
            .when(col('views') < 100000, 'Medium')
            .when(col('views') < 1000000, 'High')
            .otherwise('Viral')
        )
        
        logger.info("Data transformation completed")
        return df
    
    def save_to_s3(self, df: DataFrame, target_path: str, partition_columns: list = None) -> bool:
        """
        Save DataFrame to S3 as Parquet with partitioning.
        
        Args:
            df (DataFrame): DataFrame to save
            target_path (str): Target S3 path
            partition_columns (list): Columns to partition by
            
        Returns:
            bool: True if save was successful
        """
        logger.info(f"Saving data to S3: {target_path}")
        
        try:
            # Configure writer
            writer = df.write.mode('overwrite').format('parquet').option('compression', 'snappy')
            
            # Add partitioning if specified
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
                logger.info(f"Partitioning by: {partition_columns}")
            
            # Write to S3
            writer.save(target_path)
            
            logger.info(f"Successfully saved {df.count()} records to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to S3: {e}")
            return False
    
    def register_in_glue_catalog(self, s3_path: str, table_name: str, database_name: str) -> bool:
        """
        Register the data in AWS Glue Data Catalog.
        
        Args:
            s3_path (str): S3 path to the data
            table_name (str): Table name
            database_name (str): Database name
            
        Returns:
            bool: True if registration was successful
        """
        logger.info(f"Registering table {table_name} in database {database_name}")
        
        try:
            # Read the data to infer schema
            df = self.spark_session.read.parquet(s3_path)
            
            # Convert to DynamicFrame
            dynamic_frame = DynamicFrame.fromDF(df, self.glue_context, table_name)
            
            # Write to Glue Catalog
            sink = self.glue_context.getSink(
                connection_type="s3",
                path=s3_path,
                enableUpdateCatalog=True,
                updateBehavior="UPDATE_IN_DATABASE",
                partitionKeys=["region_code", "category_id"],
                database=database_name,
                table=table_name
            )
            
            sink.setFormat("glueparquet")
            sink.setCatalogInfo(catalogDatabase=database_name, catalogTableName=table_name)
            sink.writeFrame(dynamic_frame)
            
            logger.info(f"Successfully registered table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering table in Glue Catalog: {e}")
            return False
    
    def run_etl_pipeline(self) -> bool:
        """
        Run the complete ETL pipeline.
        
        Returns:
            bool: True if pipeline was successful
        """
        try:
            logger.info("Starting ETL pipeline")
            
            # Build source path
            source_path = f"s3://{self.args['SOURCE_BUCKET']}/"
            
            # Load data
            df = self.load_data_from_s3(source_path)
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Transform data
            df_transformed = self.transform_data(df_clean)
            
            # Build target path
            target_path = f"s3://{self.args['TARGET_BUCKET']}/youtube_cleaned/"
            
            # Save to S3 with partitioning
            partition_columns = ['region_code', 'category_id']
            success = self.save_to_s3(df_transformed, target_path, partition_columns)
            
            if not success:
                return False
            
            # Register in Glue Catalog
            catalog_success = self.register_in_glue_catalog(
                target_path,
                self.args['TABLE_NAME'],
                self.args['DATABASE_NAME']
            )
            
            if catalog_success:
                logger.info("ETL pipeline completed successfully")
            else:
                logger.warning("ETL pipeline completed but catalog registration failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in ETL pipeline: {e}")
            return False
    
    def commit_job(self):
        """Commit the Glue job."""
        try:
            self.job.commit()
            logger.info("Glue job committed successfully")
        except Exception as e:
            logger.error(f"Error committing job: {e}")
            raise


def main():
    """
    Main function to run the Glue ETL job.
    """
    try:
        # Initialize ETL job
        etl = YouTubeGlueETL()
        
        # Run pipeline
        success = etl.run_etl_pipeline()
        
        if success:
            logger.info("ETL job completed successfully")
        else:
            logger.error("ETL job failed")
            sys.exit(1)
        
        # Commit job
        etl.commit_job()
        
    except Exception as e:
        logger.error(f"Fatal error in main function: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
