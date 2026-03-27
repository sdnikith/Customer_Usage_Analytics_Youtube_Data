"""
Data Validation Utilities

This module provides utility functions for data validation using both Great Expectations
and Pandas-based validation as fallback for local testing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class PandasDataValidator:
    """
    Pandas-based data validator as fallback for local testing.
    
    Attributes:
        validation_rules (Dict): Validation rules configuration
        results (Dict): Validation results
    """
    
    def __init__(self):
        """Initialize the validator with default rules."""
        self.validation_rules = self._get_default_rules()
        self.results = {
            'total_expectations': 0,
            'successful_expectations': 0,
            'failed_expectations': 0,
            'details': [],
            'quality_score': 0.0,
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("PandasDataValidator initialized")
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """
        Get default validation rules.
        
        Returns:
            Dict[str, Any]: Default validation rules
        """
        return {
            'required_columns': [
                'video_id', 'title', 'channel_title', 'views', 'likes', 
                'comments', 'category_id', 'region_code'
            ],
            'non_null_columns': [
                'video_id', 'title', 'channel_title', 'views', 'likes'
            ],
            'unique_columns': ['video_id'],
            'category_id_range': (1, 44),
            'numeric_ranges': {
                'views': (0, 10**12),
                'likes': (0, 10**11),
                'comments': (0, 10**10)
            },
            'string_length_ranges': {
                'title': (1, 200),
                'description': (0, 5000)
            },
            'row_count_min': 100000,
            'mean_views_min': 1000,
            'region_codes': ['US', 'GB', 'CA', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU'],
            'engagement_rate_range': (0, 1),
            'expected_types': {
                'video_id': 'object',
                'title': 'object',
                'channel_title': 'object',
                'views': 'int64',
                'likes': 'int64',
                'comments': 'int64',
                'category_id': 'int64',
                'region_code': 'object'
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive validation on DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info(f"Starting validation on DataFrame with {len(df)} rows")
        
        # Reset results
        self.results = {
            'total_expectations': 0,
            'successful_expectations': 0,
            'failed_expectations': 0,
            'details': [],
            'quality_score': 0.0,
            'validation_timestamp': datetime.now(timezone.utc).isoformat(),
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        # Run all validations
        self._validate_required_columns(df)
        self._validate_non_null_columns(df)
        self._validate_unique_columns(df)
        self._validate_category_id_range(df)
        self._validate_numeric_ranges(df)
        self._validate_string_lengths(df)
        self._validate_row_count(df)
        self._validate_mean_views(df)
        self._validate_region_codes(df)
        self._validate_data_types(df)
        self._validate_engagement_rate(df)
        self._validate_business_rules(df)
        
        # Calculate quality score
        self._calculate_quality_score()
        
        logger.info(f"Validation completed. Quality score: {self.results['quality_score']:.2f}%")
        return self.results
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present."""
        columns = df.columns.tolist()
        missing_columns = []
        
        for col in self.validation_rules['required_columns']:
            self.results['total_expectations'] += 1
            if col in columns:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': f'expect_column_to_exist',
                    'column': col,
                    'success': True,
                    'message': f'Column {col} exists'
                })
            else:
                self.results['failed_expectations'] += 1
                missing_columns.append(col)
                self.results['details'].append({
                    'expectation': 'expect_column_to_exist',
                    'column': col,
                    'success': False,
                    'message': f'Column {col} is missing'
                })
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
    
    def _validate_non_null_columns(self, df: pd.DataFrame) -> None:
        """Validate that specified columns have no null values."""
        for col in self.validation_rules['non_null_columns']:
            if col not in df.columns:
                continue
                
            self.results['total_expectations'] += 1
            null_count = df[col].isnull().sum()
            
            if null_count == 0:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_not_be_null',
                    'column': col,
                    'success': True,
                    'message': f'Column {col} has no null values'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_not_be_null',
                    'column': col,
                    'success': False,
                    'message': f'Column {col} has {null_count} null values'
                })
    
    def _validate_unique_columns(self, df: pd.DataFrame) -> None:
        """Validate that specified columns have unique values."""
        for col in self.validation_rules['unique_columns']:
            if col not in df.columns:
                continue
                
            self.results['total_expectations'] += 1
            duplicate_count = df[col].duplicated().sum()
            
            if duplicate_count == 0:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_unique',
                    'column': col,
                    'success': True,
                    'message': f'Column {col} has all unique values'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_unique',
                    'column': col,
                    'success': False,
                    'message': f'Column {col} has {duplicate_count} duplicate values'
                })
    
    def _validate_category_id_range(self, df: pd.DataFrame) -> None:
        """Validate category_id is within valid range."""
        if 'category_id' not in df.columns:
            return
            
        self.results['total_expectations'] += 1
        min_val, max_val = self.validation_rules['category_id_range']
        invalid_count = ((df['category_id'] < min_val) | (df['category_id'] > max_val)).sum()
        
        if invalid_count == 0:
            self.results['successful_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_in_set',
                'column': 'category_id',
                'success': True,
                'message': f'All category_id values are within range {min_val}-{max_val}'
            })
        else:
            self.results['failed_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_in_set',
                'column': 'category_id',
                'success': False,
                'message': f'{invalid_count} category_id values are outside range {min_val}-{max_val}'
            })
    
    def _validate_numeric_ranges(self, df: pd.DataFrame) -> None:
        """Validate numeric columns are within expected ranges."""
        for col, (min_val, max_val) in self.validation_rules['numeric_ranges'].items():
            if col not in df.columns:
                continue
                
            self.results['total_expectations'] += 1
            invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
            
            if invalid_count == 0:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_between',
                    'column': col,
                    'success': True,
                    'message': f'All {col} values are within range {min_val}-{max_val}'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_between',
                    'column': col,
                    'success': False,
                    'message': f'{invalid_count} {col} values are outside range {min_val}-{max_val}'
                })
    
    def _validate_string_lengths(self, df: pd.DataFrame) -> None:
        """Validate string column lengths."""
        for col, (min_len, max_len) in self.validation_rules['string_length_ranges'].items():
            if col not in df.columns:
                continue
                
            self.results['total_expectations'] += 1
            
            # Convert to string and handle nulls
            str_lengths = df[col].astype(str).str.len()
            invalid_count = ((str_lengths < min_len) | (str_lengths > max_len)).sum()
            
            if invalid_count == 0:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_value_lengths_to_be_between',
                    'column': col,
                    'success': True,
                    'message': f'All {col} lengths are within range {min_len}-{max_len}'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_value_lengths_to_be_between',
                    'column': col,
                    'success': False,
                    'message': f'{invalid_count} {col} lengths are outside range {min_len}-{max_len}'
                })
    
    def _validate_row_count(self, df: pd.DataFrame) -> None:
        """Validate minimum row count."""
        self.results['total_expectations'] += 1
        min_rows = self.validation_rules['row_count_min']
        
        if len(df) >= min_rows:
            self.results['successful_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_table_row_count_to_be_between',
                'column': None,
                'success': True,
                'message': f'Row count {len(df)} meets minimum requirement of {min_rows}'
            })
        else:
            self.results['failed_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_table_row_count_to_be_between',
                'column': None,
                'success': False,
                'message': f'Row count {len(df)} is below minimum requirement of {min_rows}'
            })
    
    def _validate_mean_views(self, df: pd.DataFrame) -> None:
        """Validate minimum mean views."""
        if 'views' not in df.columns:
            return
            
        self.results['total_expectations'] += 1
        min_mean = self.validation_rules['mean_views_min']
        actual_mean = df['views'].mean()
        
        if actual_mean >= min_mean:
            self.results['successful_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_mean_to_be_between',
                'column': 'views',
                'success': True,
                'message': f'Mean views {actual_mean:.2f} meets minimum requirement of {min_mean}'
            })
        else:
            self.results['failed_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_mean_to_be_between',
                'column': 'views',
                'success': False,
                'message': f'Mean views {actual_mean:.2f} is below minimum requirement of {min_mean}'
            })
    
    def _validate_region_codes(self, df: pd.DataFrame) -> None:
        """Validate region codes are in expected set."""
        if 'region_code' not in df.columns:
            return
            
        self.results['total_expectations'] += 1
        valid_regions = self.validation_rules['region_codes']
        invalid_count = ~df['region_code'].isin(valid_regions).sum()
        
        if invalid_count == 0:
            self.results['successful_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_in_set',
                'column': 'region_code',
                'success': True,
                'message': f'All region codes are in the expected set'
            })
        else:
            self.results['failed_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_in_set',
                'column': 'region_code',
                'success': False,
                'message': f'{invalid_count} region codes are not in the expected set'
            })
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate column data types."""
        for col, expected_type in self.validation_rules['expected_types'].items():
            if col not in df.columns:
                continue
                
            self.results['total_expectations'] += 1
            actual_type = str(df[col].dtype)
            
            if actual_type == expected_type:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_of_type',
                    'column': col,
                    'success': True,
                    'message': f'Column {col} has expected type {expected_type}'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_of_type',
                    'column': col,
                    'success': False,
                    'message': f'Column {col} has type {actual_type}, expected {expected_type}'
                })
    
    def _validate_engagement_rate(self, df: pd.DataFrame) -> None:
        """Validate engagement rate is within valid range."""
        if 'engagement_rate' not in df.columns:
            return
            
        self.results['total_expectations'] += 1
        min_val, max_val = self.validation_rules['engagement_rate_range']
        invalid_count = ((df['engagement_rate'] < min_val) | (df['engagement_rate'] > max_val)).sum()
        
        if invalid_count == 0:
            self.results['successful_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_between',
                'column': 'engagement_rate',
                'success': True,
                'message': f'All engagement_rate values are within range {min_val}-{max_val}'
            })
        else:
            self.results['failed_expectations'] += 1
            self.results['details'].append({
                'expectation': 'expect_column_values_to_be_between',
                'column': 'engagement_rate',
                'success': False,
                'message': f'{invalid_count} engagement_rate values are outside range {min_val}-{max_val}'
            })
    
    def _validate_business_rules(self, df: pd.DataFrame) -> None:
        """Validate business-specific rules."""
        # Rule: likes should generally be greater than comments
        if 'likes' in df.columns and 'comments' in df.columns:
            self.results['total_expectations'] += 1
            violation_count = (df['likes'] < df['comments']).sum()
            
            # Allow some violations (up to 10%)
            violation_rate = violation_count / len(df)
            
            if violation_rate <= 0.1:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_pair_values_A_to_be_greater_than_B',
                    'column': 'likes_vs_comments',
                    'success': True,
                    'message': f'Likes > Comments rule satisfied (violation rate: {violation_rate:.2%})'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_pair_values_A_to_be_greater_than_B',
                    'column': 'likes_vs_comments',
                    'success': False,
                    'message': f'Likes > Comments rule violated (violation rate: {violation_rate:.2%})'
                })
        
        # Rule: trending_date should be after publish_date
        if 'publish_date' in df.columns and 'trending_date' in df.columns:
            self.results['total_expectations'] += 1
            
            # Convert to datetime if needed
            publish_dates = pd.to_datetime(df['publish_date'], errors='coerce')
            trending_dates = pd.to_datetime(df['trending_date'], errors='coerce')
            
            violation_count = (trending_dates < publish_dates).sum()
            
            if violation_count == 0:
                self.results['successful_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_increasing',
                    'column': 'trending_date_vs_publish_date',
                    'success': True,
                    'message': 'All trending dates are after publish dates'
                })
            else:
                self.results['failed_expectations'] += 1
                self.results['details'].append({
                    'expectation': 'expect_column_values_to_be_increasing',
                    'column': 'trending_date_vs_publish_date',
                    'success': False,
                    'message': f'{violation_count} trending dates are before publish dates'
                })
    
    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score."""
        if self.results['total_expectations'] == 0:
            self.results['quality_score'] = 0.0
        else:
            self.results['quality_score'] = (
                self.results['successful_expectations'] / self.results['total_expectations']
            ) * 100
        
        self.results['quality_score'] = round(self.results['quality_score'], 2)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Returns:
            Dict[str, Any]: Validation summary
        """
        return {
            'quality_score': self.results['quality_score'],
            'total_expectations': self.results['total_expectations'],
            'successful_expectations': self.results['successful_expectations'],
            'failed_expectations': self.results['failed_expectations'],
            'success_rate': (
                self.results['successful_expectations'] / self.results['total_expectations'] * 100
                if self.results['total_expectations'] > 0 else 0
            ),
            'validation_timestamp': self.results['validation_timestamp']
        }
    
    def get_failed_expectations(self) -> List[Dict[str, Any]]:
        """
        Get list of failed expectations.
        
        Returns:
            List[Dict[str, Any]]: Failed expectations
        """
        return [detail for detail in self.results['details'] if not detail['success']]
    
    def export_results(self, file_path: str) -> None:
        """
        Export validation results to JSON file.
        
        Args:
            file_path (str): Output file path
        """
        import json
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Validation results exported to {file_path}")


def create_sample_data() -> pd.DataFrame:
    """
    Create sample data for testing validation.
    
    Returns:
        pd.DataFrame: Sample YouTube data
    """
    np.random.seed(42)
    
    data = {
        'video_id': [f'video_{i}' for i in range(1000)],
        'title': [f'Test Video {i}' for i in range(1000)],
        'description': [f'This is test video number {i} with some description text' for i in range(1000)],
        'tags': [f'tag{i}|tag{i+1}|tag{i+2}' for i in range(1000)],
        'category_id': np.random.randint(1, 45, 1000),
        'channel_title': [f'Channel {i%50}' for i in range(1000)],
        'publish_date': pd.date_range('2023-01-01', periods=1000, freq='H'),
        'trending_date': pd.date_range('2023-01-02', periods=1000, freq='H'),
        'region_code': np.random.choice(['US', 'GB', 'CA', 'DE', 'FR'], 1000),
        'views': np.random.randint(1000, 1000000, 1000),
        'likes': np.random.randint(10, 10000, 1000),
        'comments': np.random.randint(1, 1000, 1000),
        'engagement_rate': np.random.uniform(0.001, 0.1, 1000),
        'title_length': [len(f'Test Video {i}') for i in range(1000)],
        'description_length': [len(f'This is test video number {i} with some description text') for i in range(1000)],
        'tag_count': [3] * 1000,
        'title_word_count': [3] * 1000
    }
    
    return pd.DataFrame(data)


def main():
    """
    Main function to test the validator.
    """
    try:
        # Create sample data
        df = create_sample_data()
        
        # Add some quality issues for testing
        df.loc[0:10, 'views'] = -100  # Negative views
        df.loc[20:30, 'category_id'] = 50  # Invalid category
        df.loc[40:45, 'video_id'] = 'duplicate_id'  # Duplicate video_id
        df.loc[50:55, 'title'] = ''  # Empty title
        df.loc[60:65, 'region_code'] = 'XX'  # Invalid region
        
        # Initialize validator
        validator = PandasDataValidator()
        
        # Run validation
        results = validator.validate_dataframe(df)
        
        # Print summary
        summary = validator.get_summary()
        print("Validation Summary:")
        print(json.dumps(summary, indent=2))
        
        # Print failed expectations
        failed = validator.get_failed_expectations()
        if failed:
            print("\nFailed Expectations:")
            for expectation in failed:
                print(f"- {expectation['expectation']}: {expectation['message']}")
        
        # Export results
        validator.export_results('data/sample/validation_results.json')
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    import json
    main()
