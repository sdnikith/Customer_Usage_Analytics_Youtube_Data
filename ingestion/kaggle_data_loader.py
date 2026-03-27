"""
Kaggle YouTube Trending Dataset Loader

This module loads and processes the Kaggle YouTube Trending dataset
including CSV files and JSON category data for multiple regions.
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleDataLoader:
    """
    Loads and processes Kaggle YouTube Trending dataset.
    
    Attributes:
        data_dir (Path): Directory containing the Kaggle dataset
        supported_regions (List[str]): List of supported region codes
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize Kaggle data loader.
        
        Args:
            data_dir (Optional[str]): Directory containing Kaggle dataset. 
                                    If None, uses current directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.supported_regions = ['US', 'GB', 'CA', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
        
        logger.info(f"KaggleDataLoader initialized with data directory: {self.data_dir}")
    
    def load_region_data(self, region_code: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Load video data and category mapping for a specific region.
        
        Args:
            region_code (str): Two-letter ISO country code
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: Video data DataFrame and category mapping
            
        Raises:
            FileNotFoundError: If data files for the region are not found
            ValueError: If region code is not supported
        """
        if region_code not in self.supported_regions:
            raise ValueError(f"Region {region_code} not supported. Supported regions: {self.supported_regions}")
        
        # File paths
        csv_file = self.data_dir / f"{region_code}videos.csv"
        json_file = self.data_dir / f"{region_code}_category_id.json"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        logger.info(f"Loading data for region {region_code}")
        
        # Load CSV data
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} records from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            raise
        
        # Load category mapping
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                category_data = json.load(f)
            
            # Extract category mapping
            category_mapping = {}
            for item in category_data.get('items', []):
                category_id = item.get('id')
                category_title = item.get('snippet', {}).get('title')
                if category_id and category_title:
                    category_mapping[category_id] = category_title
            
            logger.info(f"Loaded {len(category_mapping)} categories from {json_file}")
            
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file}: {e}")
            raise
        
        return df, category_mapping
    
    def load_all_regions(self) -> Dict[str, Tuple[pd.DataFrame, Dict[str, str]]]:
        """
        Load data for all supported regions.
        
        Returns:
            Dict[str, Tuple[pd.DataFrame, Dict[str, str]]]: 
                Dictionary mapping region codes to (DataFrame, category_mapping)
        """
        logger.info("Loading data for all supported regions")
        
        all_data = {}
        for region in self.supported_regions:
            try:
                df, categories = self.load_region_data(region)
                all_data[region] = (df, categories)
                logger.info(f"Successfully loaded data for region {region}")
            except FileNotFoundError as e:
                logger.warning(f"Skipping region {region}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading region {region}: {e}")
                continue
        
        logger.info(f"Loaded data for {len(all_data)} regions")
        return all_data
    
    def standardize_dataframe(self, df: pd.DataFrame, region_code: str, 
                            category_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Standardize DataFrame to match the expected schema.
        
        Args:
            df (pd.DataFrame): Raw DataFrame from Kaggle
            region_code (str): Region code
            category_mapping (Dict[str, str]): Category ID to name mapping
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        logger.info(f"Standardizing DataFrame for region {region_code}")
        
        # Create a copy to avoid modifying the original
        standardized_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'video_id': 'video_id',
            'title': 'title',
            'channel_title': 'channel_title',
            'category_id': 'category_id',
            'publish_time': 'publish_date',
            'tags': 'tags',
            'views': 'views',
            'likes': 'likes',
            'dislikes': 'dislikes',
            'comment_count': 'comments',
            'thumbnail_link': 'thumbnail_url',
            'description': 'description'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in standardized_df.columns:
                standardized_df = standardized_df.rename(columns={old_name: new_name})
        
        # Add region code
        standardized_df['region_code'] = region_code
        
        # Add category name
        standardized_df['category_name'] = standardized_df['category_id'].astype(str).map(category_mapping)
        
        # Convert data types
        numeric_columns = ['views', 'likes', 'comments', 'category_id']
        for col in numeric_columns:
            if col in standardized_df.columns:
                standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce').fillna(0).astype(int)
        
        # Parse dates
        if 'publish_date' in standardized_df.columns:
            standardized_df['publish_date'] = pd.to_datetime(
                standardized_df['publish_date'], 
                errors='coerce',
                utc=True
            )
        
        # Add trending date (use current date as proxy)
        standardized_df['trending_date'] = datetime.now(timezone.utc)
        
        # Process tags
        if 'tags' in standardized_df.columns:
            standardized_df['tags'] = standardized_df['tags'].fillna('')
            standardized_df['tag_count'] = standardized_df['tags'].apply(
                lambda x: len(x.split('|')) if x and x != '[none]' else 0
            )
        
        # Add calculated fields
        standardized_df['engagement_rate'] = (
            (standardized_df['likes'] + standardized_df['comments']) / 
            standardized_df['views'].replace(0, 1)
        )
        
        standardized_df['title_length'] = standardized_df['title'].str.len()
        standardized_df['description_length'] = standardized_df['description'].str.len().fillna(0)
        standardized_df['title_word_count'] = standardized_df['title'].str.split().str.len()
        
        # Add extraction timestamp
        standardized_df['extracted_at'] = datetime.now(timezone.utc)
        
        # Select and reorder columns
        expected_columns = [
            'video_id', 'title', 'description', 'tags', 'tag_count',
            'category_id', 'category_name', 'channel_id', 'channel_title',
            'publish_date', 'trending_date', 'region_code', 'views', 'likes',
            'comments', 'engagement_rate', 'title_length', 'description_length',
            'title_word_count', 'thumbnail_url', 'extracted_at'
        ]
        
        # Add missing columns with default values
        for col in expected_columns:
            if col not in standardized_df.columns:
                if col == 'channel_id':
                    standardized_df[col] = ''
                elif col in ['title_length', 'description_length', 'tag_count', 'title_word_count']:
                    standardized_df[col] = 0
                elif col == 'engagement_rate':
                    standardized_df[col] = 0.0
                else:
                    standardized_df[col] = None
        
        # Select only expected columns
        standardized_df = standardized_df[expected_columns]
        
        logger.info(f"Standardized DataFrame has {len(standardized_df)} rows and {len(standardized_df.columns)} columns")
        return standardized_df
    
    def combine_all_regions(self) -> pd.DataFrame:
        """
        Load and combine data from all available regions.
        
        Returns:
            pd.DataFrame: Combined DataFrame with data from all regions
        """
        logger.info("Combining data from all regions")
        
        all_dataframes = []
        
        region_data = self.load_all_regions()
        
        for region_code, (df, categories) in region_data.items():
            try:
                standardized_df = self.standardize_dataframe(df, region_code, categories)
                all_dataframes.append(standardized_df)
                logger.info(f"Added {len(standardized_df)} records from region {region_code}")
            except Exception as e:
                logger.error(f"Error processing region {region_code}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No data could be loaded from any region")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates based on video_id
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['video_id'], keep='last')
        final_count = len(combined_df)
        
        logger.info(f"Combined dataset: {final_count} unique videos (removed {initial_count - final_count} duplicates)")
        
        return combined_df
    
    def save_sample_data(self, output_dir: str = "data/sample", num_samples: int = 1000) -> None:
        """
        Save sample data for testing and development.
        
        Args:
            output_dir (str): Output directory for sample data
            num_samples (int): Number of samples to save
        """
        logger.info(f"Saving sample data to {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load combined data
            combined_df = self.combine_all_regions()
            
            # Sample data
            if len(combined_df) > num_samples:
                sample_df = combined_df.sample(n=num_samples, random_state=42)
            else:
                sample_df = combined_df
            
            # Save as CSV
            csv_file = output_path / "sample_videos.csv"
            sample_df.to_csv(csv_file, index=False)
            logger.info(f"Saved {len(sample_df)} samples to {csv_file}")
            
            # Save category mappings
            region_data = self.load_all_regions()
            all_categories = {}
            for region, (_, categories) in region_data.items():
                all_categories[region] = categories
            
            json_file = output_path / "sample_categories.json"
            with open(json_file, 'w') as f:
                json.dump(all_categories, f, indent=2)
            logger.info(f"Saved category mappings to {json_file}")
            
        except Exception as e:
            logger.error(f"Error saving sample data: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dict[str, any]: Summary statistics
        """
        logger.info("Generating data summary")
        
        try:
            combined_df = self.combine_all_regions()
            
            summary = {
                'total_videos': len(combined_df),
                'unique_channels': combined_df['channel_title'].nunique(),
                'regions': combined_df['region_code'].unique().tolist(),
                'date_range': {
                    'start': combined_df['publish_date'].min().isoformat() if pd.notna(combined_df['publish_date'].min()) else None,
                    'end': combined_df['publish_date'].max().isoformat() if pd.notna(combined_df['publish_date'].max()) else None
                },
                'statistics': {
                    'total_views': int(combined_df['views'].sum()),
                    'avg_views': float(combined_df['views'].mean()),
                    'max_views': int(combined_df['views'].max()),
                    'total_likes': int(combined_df['likes'].sum()),
                    'avg_likes': float(combined_df['likes'].mean()),
                    'total_comments': int(combined_df['comments'].sum()),
                    'avg_comments': float(combined_df['comments'].mean()),
                    'avg_engagement_rate': float(combined_df['engagement_rate'].mean())
                },
                'categories': combined_df['category_id'].value_counts().to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            raise


def main():
    """
    Main function to test the Kaggle data loader.
    """
    try:
        loader = KaggleDataLoader()
        
        # Test loading all regions
        combined_df = loader.combine_all_regions()
        print(f"Combined dataset shape: {combined_df.shape}")
        
        # Get summary
        summary = loader.get_data_summary()
        print("Data Summary:")
        print(json.dumps(summary, indent=2))
        
        # Save sample data
        loader.save_sample_data()
        print("Sample data saved successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
