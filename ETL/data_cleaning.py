"""
Data Cleaning Utilities for YouTube Analytics

This module provides comprehensive data cleaning functions for YouTube data
including validation, standardization, and quality checks.
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class YouTubeDataCleaner:
    """
    Comprehensive data cleaning utilities for YouTube data.
    
    Attributes:
        validation_rules (Dict): Data validation rules
        cleaning_stats (Dict): Statistics about cleaning operations
    """
    
    def __init__(self):
        """Initialize the data cleaner with validation rules."""
        self.validation_rules = {
            'required_fields': ['video_id', 'title', 'channel_title', 'views', 'likes'],
            'numeric_fields': ['views', 'likes', 'comments', 'tag_count', 'category_id'],
            'text_fields': ['title', 'description', 'tags', 'channel_title'],
            'date_fields': ['publish_date', 'trending_date', 'extracted_at'],
            'category_id_range': (1, 44),
            'min_title_length': 1,
            'max_title_length': 200,
            'min_description_length': 0,
            'max_description_length': 5000,
            'min_views': 0,
            'max_views': 10**12,  # 1 trillion views
            'min_likes': 0,
            'max_likes': 10**11,  # 100 billion likes
            'min_comments': 0,
            'max_comments': 10**10  # 10 billion comments
        }
        
        self.cleaning_stats = {
            'initial_records': 0,
            'final_records': 0,
            'duplicates_removed': 0,
            'nulls_removed': 0,
            'invalid_values_corrected': 0,
            'outliers_handled': 0
        }
        
        logger.info("YouTubeDataCleaner initialized")
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema against expected structure.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        missing_fields = []
        for field in self.validation_rules['required_fields']:
            if field not in df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Check data types
        expected_types = {
            'video_id': 'object',
            'title': 'object',
            'channel_title': 'object',
            'views': 'int64',
            'likes': 'int64',
            'comments': 'int64',
            'category_id': 'int64'
        }
        
        for field, expected_type in expected_types.items():
            if field in df.columns:
                actual_type = str(df[field].dtype)
                if actual_type != expected_type and not pd.api.types.is_numeric_dtype(df[field]):
                    issues.append(f"Field {field} has type {actual_type}, expected {expected_type}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize text fields.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned text fields
        """
        logger.info("Cleaning text fields")
        
        df_clean = df.copy()
        
        for field in self.validation_rules['text_fields']:
            if field in df_clean.columns:
                # Handle nulls
                df_clean[field] = df_clean[field].fillna('')
                
                # Convert to string
                df_clean[field] = df_clean[field].astype(str)
                
                # Remove extra whitespace
                df_clean[field] = df_clean[field].str.strip()
                df_clean[field] = df_clean[field].str.replace(r'\s+', ' ', regex=True)
                
                # Remove special characters from title (keep basic punctuation)
                if field == 'title':
                    df_clean[field] = df_clean[field].str.replace(r'[^\w\s\-.,!?;:]', '', regex=True)
                
                # Handle empty descriptions
                if field == 'description':
                    df_clean[field] = df_clean[field].replace('', 'No description available')
                
                # Clean tags
                if field == 'tags':
                    df_clean[field] = df_clean[field].str.replace(r'[^\w\s\-\|,]', '', regex=True)
        
        logger.info("Text fields cleaned")
        return df_clean
    
    def clean_numeric_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate numeric fields.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned numeric fields
        """
        logger.info("Cleaning numeric fields")
        
        df_clean = df.copy()
        corrections = 0
        
        for field in self.validation_rules['numeric_fields']:
            if field in df_clean.columns:
                # Handle nulls
                df_clean[field] = df_clean[field].fillna(0)
                
                # Convert to numeric
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce').fillna(0)
                
                # Ensure non-negative values
                if field in ['views', 'likes', 'comments', 'tag_count']:
                    negative_count = (df_clean[field] < 0).sum()
                    if negative_count > 0:
                        df_clean[field] = df_clean[field].clip(lower=0)
                        corrections += negative_count
                
                # Handle category_id validation
                if field == 'category_id':
                    min_cat, max_cat = self.validation_rules['category_id_range']
                    invalid_cats = ((df_clean[field] < min_cat) | (df_clean[field] > max_cat)).sum()
                    if invalid_cats > 0:
                        df_clean[field] = df_clean[field].clip(lower=min_cat, upper=max_cat)
                        corrections += invalid_cats
        
        self.cleaning_stats['invalid_values_corrected'] += corrections
        logger.info(f"Numeric fields cleaned. {corrections} corrections made")
        return df_clean
    
    def clean_date_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize date fields.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with cleaned date fields
        """
        logger.info("Cleaning date fields")
        
        df_clean = df.copy()
        
        for field in self.validation_rules['date_fields']:
            if field in df_clean.columns:
                # Convert to datetime
                df_clean[field] = pd.to_datetime(df_clean[field], errors='coerce', utc=True)
                
                # Handle null dates
                if field == 'publish_date':
                    # Set null publish dates to a default date
                    null_count = df_clean[field].isnull().sum()
                    if null_count > 0:
                        df_clean[field] = df_clean[field].fillna(pd.Timestamp('2000-01-01', tz='UTC'))
                        logger.info(f"Filled {null_count} null publish dates with default")
                
                elif field == 'trending_date':
                    # Set null trending dates to current date
                    null_count = df_clean[field].isnull().sum()
                    if null_count > 0:
                        df_clean[field] = df_clean[field].fillna(datetime.now(timezone.utc))
                        logger.info(f"Filled {null_count} null trending dates with current date")
        
        logger.info("Date fields cleaned")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """
        Remove duplicate records based on specified columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            subset (List[str]): Columns to check for duplicates
            
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        if subset is None:
            subset = ['video_id']
        
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep='last')
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        self.cleaning_stats['duplicates_removed'] += duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicate records based on {subset}")
        
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numeric fields using IQR method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        logger.info("Handling outliers")
        
        df_clean = df.copy()
        outliers_handled = 0
        
        numeric_fields = ['views', 'likes', 'comments']
        
        for field in numeric_fields:
            if field in df_clean.columns:
                # Calculate IQR
                Q1 = df_clean[field].quantile(0.25)
                Q3 = df_clean[field].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((df_clean[field] < lower_bound) | (df_clean[field] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers at bounds
                    df_clean[field] = df_clean[field].clip(lower=lower_bound, upper=upper_bound)
                    outliers_handled += outliers
                    logger.info(f"Capped {outliers} outliers in {field}")
        
        self.cleaning_stats['outliers_handled'] += outliers_handled
        logger.info(f"Total outliers handled: {outliers_handled}")
        
        return df_clean
    
    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived fields for analysis.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with derived fields
        """
        logger.info("Adding derived fields")
        
        df_clean = df.copy()
        
        # Engagement rate
        df_clean['engagement_rate'] = np.where(
            df_clean['views'] > 0,
            (df_clean['likes'] + df_clean['comments']) / df_clean['views'],
            0.0
        )
        
        # Title metrics
        df_clean['title_length'] = df_clean['title'].str.len()
        df_clean['title_word_count'] = df_clean['title'].str.split().str.len()
        
        # Description metrics
        df_clean['description_length'] = df_clean['description'].str.len()
        
        # Tag metrics
        if 'tags' in df_clean.columns:
            df_clean['tag_count'] = df_clean['tags'].apply(
                lambda x: len(x.split('|')) if x and x != '' else 0
            )
        
        # Time to trending (in hours)
        if 'publish_date' in df_clean.columns and 'trending_date' in df_clean.columns:
            df_clean['time_to_trending_hours'] = (
                (df_clean['trending_date'] - df_clean['publish_date']).dt.total_seconds() / 3600
            )
            df_clean['time_to_trending_hours'] = df_clean['time_to_trending_hours'].clip(lower=0)
        
        # Boolean flags
        df_clean['has_description'] = df_clean['description'] != 'No description available'
        df_clean['has_tags'] = df_clean['tag_count'] > 0
        
        # Categorical fields
        df_clean['engagement_category'] = pd.cut(
            df_clean['engagement_rate'],
            bins=[-np.inf, 0.01, 0.05, np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        df_clean['view_category'] = pd.cut(
            df_clean['views'],
            bins=[-np.inf, 10000, 100000, 1000000, np.inf],
            labels=['Low', 'Medium', 'High', 'Viral']
        )
        
        logger.info("Derived fields added")
        return df_clean
    
    def comprehensive_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting comprehensive data cleaning")
        
        # Initialize stats
        self.cleaning_stats['initial_records'] = len(df)
        
        # Validate schema
        is_valid, issues = self.validate_schema(df)
        if not is_valid:
            logger.warning(f"Schema validation issues: {issues}")
        
        # Remove duplicates first
        df_clean = self.remove_duplicates(df)
        
        # Clean text fields
        df_clean = self.clean_text_fields(df_clean)
        
        # Clean numeric fields
        df_clean = self.clean_numeric_fields(df_clean)
        
        # Clean date fields
        df_clean = self.clean_date_fields(df_clean)
        
        # Handle outliers
        df_clean = self.handle_outliers(df_clean)
        
        # Add derived fields
        df_clean = self.add_derived_fields(df_clean)
        
        # Remove records with null critical fields
        critical_fields = self.validation_rules['required_fields']
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=critical_fields)
        nulls_removed = initial_count - len(df_clean)
        self.cleaning_stats['nulls_removed'] += nulls_removed
        
        # Final stats
        self.cleaning_stats['final_records'] = len(df_clean)
        
        logger.info(f"Comprehensive cleaning completed. "
                   f"Records: {self.cleaning_stats['initial_records']} -> {self.cleaning_stats['final_records']}")
        
        return df_clean
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Get a report of cleaning operations.
        
        Returns:
            Dict[str, Any]: Cleaning statistics
        """
        return {
            'cleaning_statistics': self.cleaning_stats.copy(),
            'validation_rules': self.validation_rules,
            'data_quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall data quality score.
        
        Returns:
            float: Quality score between 0 and 100
        """
        if self.cleaning_stats['initial_records'] == 0:
            return 0.0
        
        # Base score starts at 100
        score = 100.0
        
        # Deductions for issues
        total_issues = (
            self.cleaning_stats['duplicates_removed'] +
            self.cleaning_stats['nulls_removed'] +
            self.cleaning_stats['invalid_values_corrected'] +
            self.cleaning_stats['outliers_handled']
        )
        
        if total_issues > 0:
            deduction = (total_issues / self.cleaning_stats['initial_records']) * 100
            score = max(0, score - deduction)
        
        return round(score, 2)


def main():
    """
    Main function to test the data cleaner.
    """
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'video_id': ['test1', 'test2', 'test3', 'test1'],  # duplicate
            'title': ['Test Video 1', 'Test Video 2', '', 'Test Video 1'],
            'description': ['Desc 1', None, 'Desc 3', 'Desc 1'],
            'tags': ['tag1|tag2', '', 'tag3', 'tag1|tag2'],
            'category_id': [1, 45, 2, 1],  # invalid category_id
            'channel_title': ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 1'],
            'views': [1000, -100, 300, 1000],  # negative views
            'likes': [100, 200, -50, 100],  # negative likes
            'comments': [10, 20, 30, 10],
            'publish_date': ['2023-01-01', 'invalid_date', '2023-01-03', '2023-01-01'],
            'trending_date': ['2023-01-02', '2023-01-03', None, '2023-01-02']
        })
        
        # Initialize cleaner
        cleaner = YouTubeDataCleaner()
        
        # Clean data
        cleaned_data = cleaner.comprehensive_clean(sample_data)
        
        # Get report
        report = cleaner.get_cleaning_report()
        
        print("Original data shape:", sample_data.shape)
        print("Cleaned data shape:", cleaned_data.shape)
        print("\nCleaning Report:")
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    import json
    main()
