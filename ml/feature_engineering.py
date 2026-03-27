"""
Feature Engineering Module for YouTube Analytics

This module creates engineered features for machine learning models including
temporal features, text metrics, engagement features, and categorical encodings.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeFeatureEngineer:
    """
    Feature engineering for YouTube analytics data.
    
    Attributes:
        scaler: Fitted scaler for numeric features
        encoder: Fitted encoder for categorical features
        feature_columns: List of engineered feature names
        target_columns: List of target variable names
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = None
        self.encoder = None
        self.feature_columns = []
        self.target_columns = ['views', 'engagement_rate']
        
        # YouTube category mapping for business logic
        self.category_mapping = {
            1: "Film & Animation",
            2: "Autos & Vehicles", 
            10: "Music",
            15: "Pets & Animals",
            17: "Sports",
            19: "Travel & Events",
            20: "Gaming",
            22: "People & Blogs",
            23: "Comedy",
            24: "Entertainment",
            25: "News & Politics",
            26: "Howto & Style",
            27: "Education",
            28: "Science & Technology",
            29: "Nonprofits & Activism"
        }
        
        logger.info("YouTubeFeatureEngineer initialized")
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from publish and trending dates.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with temporal features
        """
        logger.info("Creating temporal features")
        
        df_fe = df.copy()
        
        # Ensure datetime columns
        if 'publish_date' in df_fe.columns:
            df_fe['publish_date'] = pd.to_datetime(df_fe['publish_date'], errors='coerce')
        
        if 'trending_date' in df_fe.columns:
            df_fe['trending_date'] = pd.to_datetime(df_fe['trending_date'], errors='coerce')
        
        # Extract time components from publish_date
        if 'publish_date' in df_fe.columns:
            df_fe['publish_hour'] = df_fe['publish_date'].dt.hour
            df_fe['publish_day_of_week'] = df_fe['publish_date'].dt.dayofweek  # 0=Monday, 6=Sunday
            df_fe['publish_day_of_month'] = df_fe['publish_date'].dt.day
            df_fe['publish_week'] = df_fe['publish_date'].dt.isocalendar().week
            df_fe['publish_month'] = df_fe['publish_date'].dt.month
            df_fe['publish_quarter'] = df_fe['publish_date'].dt.quarter
            df_fe['publish_year'] = df_fe['publish_date'].dt.year
            
            # Cyclical features for better seasonality capture
            df_fe['publish_hour_sin'] = np.sin(2 * np.pi * df_fe['publish_hour'] / 24)
            df_fe['publish_hour_cos'] = np.cos(2 * np.pi * df_fe['publish_hour'] / 24)
            df_fe['publish_day_sin'] = np.sin(2 * np.pi * df_fe['publish_day_of_week'] / 7)
            df_fe['publish_day_cos'] = np.cos(2 * np.pi * df_fe['publish_day_of_week'] / 7)
            df_fe['publish_month_sin'] = np.sin(2 * np.pi * df_fe['publish_month'] / 12)
            df_fe['publish_month_cos'] = np.cos(2 * np.pi * df_fe['publish_month'] / 12)
            
            # Weekend flag
            df_fe['is_weekend'] = (df_fe['publish_day_of_week'] >= 5).astype(int)
            
            # Business hours flag (9 AM - 5 PM)
            df_fe['is_business_hours'] = ((df_fe['publish_hour'] >= 9) & 
                                       (df_fe['publish_hour'] <= 17)).astype(int)
            
            # Prime time flag (7 PM - 11 PM)
            df_fe['is_prime_time'] = ((df_fe['publish_hour'] >= 19) & 
                                     (df_fe['publish_hour'] <= 23)).astype(int)
        
        # Calculate time to trending
        if 'publish_date' in df_fe.columns and 'trending_date' in df_fe.columns:
            df_fe['time_to_trending_hours'] = (
                (df_fe['trending_date'] - df_fe['publish_date']).dt.total_seconds() / 3600
            )
            df_fe['time_to_trending_hours'] = df_fe['time_to_trending_hours'].clip(lower=0)
            
            # Time to trending categories
            df_fe['trending_speed'] = pd.cut(
                df_fe['time_to_trending_hours'],
                bins=[-np.inf, 24, 72, 168, np.inf],
                labels=['Very Fast', 'Fast', 'Medium', 'Slow']
            )
        
        logger.info(f"Created {len([col for col in df_fe.columns if col not in df.columns])} temporal features")
        return df_fe
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create text-based features from title, description, and tags.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with text features
        """
        logger.info("Creating text features")
        
        df_fe = df.copy()
        
        # Title features
        if 'title' in df_fe.columns:
            df_fe['title'] = df_fe['title'].fillna('').astype(str)
            df_fe['title_length'] = df_fe['title'].str.len()
            df_fe['title_word_count'] = df_fe['title'].str.split().str.len()
            df_fe['title_char_per_word'] = np.where(
                df_fe['title_word_count'] > 0,
                df_fe['title_length'] / df_fe['title_word_count'],
                0
            )
            
            # Title special features
            df_fe['title_has_numbers'] = df_fe['title'].str.contains(r'\d').astype(int)
            df_fe['title_has_uppercase'] = df_fe['title'].str.contains(r'[A-Z]').astype(int)
            df_fe['title_exclamation_count'] = df_fe['title'].str.count(r'!')
            df_fe['title_question_count'] = df_fe['title'].str.count(r'\?')
            df_fe['title_has_clickbait_words'] = df_fe['title'].str.contains(
                r'\b(shock|amazing|incredible|unbelievable|you won\'t believe|secret|revealed)\b',
                case=False, regex=True
            ).astype(int)
        
        # Description features
        if 'description' in df_fe.columns:
            df_fe['description'] = df_fe['description'].fillna('').astype(str)
            df_fe['description_length'] = df_fe['description'].str.len()
            df_fe['description_word_count'] = df_fe['description'].str.split().str.len()
            df_fe['description_char_per_word'] = np.where(
                df_fe['description_word_count'] > 0,
                df_fe['description_length'] / df_fe['description_word_count'],
                0
            )
            
            # Description quality indicators
            df_fe['description_has_links'] = df_fe['description'].str.contains(r'http').astype(int)
            df_fe['description_has_emoji'] = df_fe['description'].str.contains(
                r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
            ).astype(int)
        
        # Tags features
        if 'tags' in df_fe.columns:
            df_fe['tags'] = df_fe['tags'].fillna('').astype(str)
            
            # Count tags (handle different delimiters)
            df_fe['tag_count'] = df_fe['tags'].str.split(r'[|,;]').str.len()
            
            # Average tag length
            def avg_tag_length(tags_str):
                if not tags_str or tags_str == '':
                    return 0
                tags = re.split(r'[|,;]', tags_str)
                tags = [tag.strip() for tag in tags if tag.strip()]
                return np.mean([len(tag) for tag in tags]) if tags else 0
            
            df_fe['avg_tag_length'] = df_fe['tags'].apply(avg_tag_length)
            
            # Tag diversity (unique tags)
            def unique_tag_count(tags_str):
                if not tags_str or tags_str == '':
                    return 0
                tags = re.split(r'[|,;]', tags_str)
                tags = [tag.strip().lower() for tag in tags if tag.strip()]
                return len(set(tags)) if tags else 0
            
            df_fe['unique_tag_count'] = df_fe['tags'].apply(unique_tag_count)
        
        # Combined text features
        if all(col in df_fe.columns for col in ['title', 'description', 'tags']):
            # Total text length
            df_fe['total_text_length'] = (
                df_fe['title_length'] + df_fe['description_length'] + 
                df_fe['tags'].str.len()
            )
            
            # Text ratio features
            df_fe['title_to_desc_ratio'] = np.where(
                df_fe['description_length'] > 0,
                df_fe['title_length'] / df_fe['description_length'],
                df_fe['title_length']
            )
            
            df_fe['desc_to_total_ratio'] = np.where(
                df_fe['total_text_length'] > 0,
                df_fe['description_length'] / df_fe['total_text_length'],
                0
            )
        
        logger.info(f"Created text features")
        return df_fe
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement-based features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engagement features
        """
        logger.info("Creating engagement features")
        
        df_fe = df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['views', 'likes', 'comments']
        for col in numeric_cols:
            if col in df_fe.columns:
                df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(0)
        
        # Engagement rate (if not already present)
        if 'engagement_rate' not in df_fe.columns and all(col in df_fe.columns for col in ['views', 'likes', 'comments']):
            df_fe['engagement_rate'] = np.where(
                df_fe['views'] > 0,
                (df_fe['likes'] + df_fe['comments']) / df_fe['views'],
                0
            )
        
        # Like ratio
        if 'views' in df_fe.columns and 'likes' in df_fe.columns:
            df_fe['like_rate'] = np.where(
                df_fe['views'] > 0,
                df_fe['likes'] / df_fe['views'],
                0
            )
        
        # Comment ratio
        if 'views' in df_fe.columns and 'comments' in df_fe.columns:
            df_fe['comment_rate'] = np.where(
                df_fe['views'] > 0,
                df_fe['comments'] / df_fe['views'],
                0
            )
        
        # Like to comment ratio
        if 'likes' in df_fe.columns and 'comments' in df_fe.columns:
            df_fe['like_to_comment_ratio'] = np.where(
                df_fe['comments'] > 0,
                df_fe['likes'] / df_fe['comments'],
                df_fe['likes']  # If no comments, use likes as ratio
            )
        
        # Engagement score (normalized)
        if all(col in df_fe.columns for col in ['views', 'likes', 'comments']):
            # Log transformation to handle skewness
            df_fe['log_views'] = np.log1p(df_fe['views'])
            df_fe['log_likes'] = np.log1p(df_fe['likes'])
            df_fe['log_comments'] = np.log1p(df_fe['comments'])
            
            # Combined engagement score
            df_fe['engagement_score'] = (
                0.5 * df_fe['log_likes'] + 
                0.3 * df_fe['log_comments'] + 
                0.2 * df_fe['engagement_rate']
            )
        
        # View categories (if not already present)
        if 'views' in df_fe.columns and 'view_category' not in df_fe.columns:
            df_fe['view_category'] = pd.cut(
                df_fe['views'],
                bins=[-np.inf, 10000, 100000, 1000000, np.inf],
                labels=['Low', 'Medium', 'High', 'Viral']
            )
        
        # Engagement categories (if not already present)
        if 'engagement_rate' in df_fe.columns and 'engagement_category' not in df_fe.columns:
            df_fe['engagement_category'] = pd.cut(
                df_fe['engagement_rate'],
                bins=[-np.inf, 0.01, 0.05, np.inf],
                labels=['Low', 'Medium', 'High']
            )
        
        logger.info("Created engagement features")
        return df_fe
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features and encodings.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with categorical features
        """
        logger.info("Creating categorical features")
        
        df_fe = df.copy()
        
        # Category features
        if 'category_id' in df_fe.columns:
            # Category name
            df_fe['category_name'] = df_fe['category_id'].map(self.category_mapping).fillna('Unknown')
            
            # Category groupings
            entertainment_categories = [1, 23, 24]  # Film, Comedy, Entertainment
            educational_categories = [27, 28]  # Education, Science & Tech
            lifestyle_categories = [2, 15, 26]  # Autos, Pets, Howto
            
            df_fe['is_entertainment'] = df_fe['category_id'].isin(entertainment_categories).astype(int)
            df_fe['is_educational'] = df_fe['category_id'].isin(educational_categories).astype(int)
            df_fe['is_lifestyle'] = df_fe['category_id'].isin(lifestyle_categories).astype(int)
        
        # Region features
        if 'region_code' in df_fe.columns:
            # Region groupings
            english_regions = ['US', 'GB', 'CA', 'AU']
            european_regions = ['DE', 'FR', 'IT', 'NL', 'ES']
            asian_regions = ['IN', 'JP', 'KR']
            
            df_fe['is_english_speaking'] = df_fe['region_code'].isin(english_regions).astype(int)
            df_fe['is_european'] = df_fe['region_code'].isin(european_regions).astype(int)
            df_fe['is_asian'] = df_fe['region_code'].isin(asian_regions).astype(int)
        
        # Boolean flags (if not already present)
        if 'description' in df_fe.columns:
            df_fe['has_description'] = (df_fe['description'].notna() & 
                                      (df_fe['description'] != '') & 
                                      (df_fe['description'] != 'No description available')).astype(int)
        
        if 'tags' in df_fe.columns:
            df_fe['has_tags'] = (df_fe['tags'].notna() & 
                                 (df_fe['tags'] != '')).astype(int)
        
        # Channel features (if channel data available)
        if 'channel_title' in df_fe.columns:
            # Channel name length
            df_fe['channel_name_length'] = df_fe['channel_title'].str.len()
            
            # Channel name word count
            df_fe['channel_word_count'] = df_fe['channel_title'].str.split().str.len()
            
            # Channel has numbers
            df_fe['channel_has_numbers'] = df_fe['channel_title'].str.contains(r'\d').astype(int)
        
        logger.info("Created categorical features")
        return df_fe
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with interaction features
        """
        logger.info("Creating interaction features")
        
        df_fe = df.copy()
        
        # Text x Category interactions
        if all(col in df_fe.columns for col in ['title_length', 'category_id']):
            df_fe['title_length_by_category'] = df_fe['title_length'] * df_fe['category_id']
        
        if all(col in df_fe.columns for col in ['tag_count', 'category_id']):
            df_fe['tag_count_by_category'] = df_fe['tag_count'] * df_fe['category_id']
        
        # Temporal x Category interactions
        if all(col in df_fe.columns for col in ['publish_hour', 'category_id']):
            df_fe['hour_by_category'] = df_fe['publish_hour'] * df_fe['category_id']
        
        # Engagement x Text interactions
        if all(col in df_fe.columns for col in ['engagement_rate', 'title_length']):
            df_fe['engagement_by_title_length'] = df_fe['engagement_rate'] * df_fe['title_length']
        
        if all(col in df_fe.columns for col in ['engagement_rate', 'tag_count']):
            df_fe['engagement_by_tag_count'] = df_fe['engagement_rate'] * df_fe['tag_count']
        
        # Region x Category interactions
        if all(col in df_fe.columns for col in ['region_code', 'category_id']):
            # Create region-category combination
            df_fe['region_category_combo'] = df_fe['region_code'] + '_' + df_fe['category_id'].astype(str)
        
        logger.info("Created interaction features")
        return df_fe
    
    def prepare_features_for_ml(self, df: pd.DataFrame, 
                            fit_transformers: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare all features for machine learning with proper scaling and encoding.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fit_transformers (bool): Whether to fit transformers or use existing ones
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (features, targets)
        """
        logger.info("Preparing features for machine learning")
        
        # Create all feature types
        df_fe = df.copy()
        df_fe = self.create_temporal_features(df_fe)
        df_fe = self.create_text_features(df_fe)
        df_fe = self.create_engagement_features(df_fe)
        df_fe = self.create_categorical_features(df_fe)
        df_fe = self.create_interaction_features(df_fe)
        
        # Define feature groups
        numeric_features = [
            'publish_hour', 'publish_day_of_week', 'publish_month', 'publish_year',
            'time_to_trending_hours', 'title_length', 'title_word_count',
            'description_length', 'description_word_count', 'tag_count',
            'avg_tag_length', 'unique_tag_count', 'total_text_length',
            'like_rate', 'comment_rate', 'like_to_comment_ratio',
            'log_views', 'log_likes', 'log_comments', 'engagement_score',
            'channel_name_length', 'channel_word_count'
        ]
        
        categorical_features = [
            'category_id', 'region_code', 'view_category', 'engagement_category',
            'trending_speed', 'region_category_combo'
        ]
        
        # Filter to only available features
        numeric_features = [col for col in numeric_features if col in df_fe.columns]
        categorical_features = [col for col in categorical_features if col in df_fe.columns]
        
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        # Prepare features DataFrame
        X = df_fe[numeric_features + categorical_features].copy()
        
        # Handle missing values
        X[numeric_features] = X[numeric_features].fillna(0)
        X[categorical_features] = X[categorical_features].fillna('Unknown')
        
        # Scale numeric features
        if numeric_features:
            if fit_transformers or self.scaler is None:
                self.scaler = StandardScaler()
                X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
            else:
                X[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # Encode categorical features
        if categorical_features:
            if fit_transformers or self.encoder is None:
                self.encoder = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
                    ],
                    remainder='passthrough'
                )
                X_encoded = self.encoder.fit_transform(X)
                feature_names = (numeric_features + 
                               list(self.encoder.named_transformers_['cat'][1].get_feature_names_out(categorical_features)))
                X = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)
            else:
                X_encoded = self.encoder.transform(X)
                feature_names = (numeric_features + 
                               list(self.encoder.named_transformers_['cat'][1].get_feature_names_out(categorical_features)))
                X = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)
        
        # Prepare targets
        y = df_fe[self.target_columns].copy()
        
        # Store feature names
        self.feature_columns = list(X.columns)
        
        logger.info(f"Final feature matrix: {X.shape}")
        logger.info(f"Target matrix: {y.shape}")
        
        return X, y
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get feature importance information for analysis.
        
        Args:
            df (pd.DataFrame): Original DataFrame
            
        Returns:
            Dict[str, Any]: Feature importance data
        """
        X, y = self.prepare_features_for_ml(df, fit_transformers=False)
        
        return {
            'feature_names': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'numeric_features': [col for col in self.feature_columns if any(numeric in col for numeric in [
                'publish_hour', 'title_length', 'description_length', 'tag_count',
                'like_rate', 'comment_rate', 'engagement_score', 'log_views'
            ])],
            'categorical_features': [col for col in self.feature_columns if any(cat in col for cat in [
                'category_id', 'region_code', 'view_category', 'engagement_category'
            ])],
            'interaction_features': [col for col in self.feature_columns if any(intx in col for intx in [
                '_by_', '_combo', 'interaction'
            ])],
            'feature_matrix_shape': X.shape,
            'target_shape': y.shape
        }
    
    def save_transformers(self, path: str) -> bool:
        """
        Save fitted transformers to disk.
        
        Args:
            path (str): Path to save transformers
            
        Returns:
            bool: True if saved successfully
        """
        try:
            import pickle
            import os
            
            os.makedirs(path, exist_ok=True)
            
            transformers = {
                'scaler': self.scaler,
                'encoder': self.encoder,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'save_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(os.path.join(path, 'feature_transformers.pkl'), 'wb') as f:
                pickle.dump(transformers, f)
            
            logger.info(f"Transformers saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving transformers: {e}")
            return False
    
    def load_transformers(self, path: str) -> bool:
        """
        Load fitted transformers from disk.
        
        Args:
            path (str): Path to load transformers from
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            import pickle
            import os
            
            transformer_file = os.path.join(path, 'feature_transformers.pkl')
            
            if not os.path.exists(transformer_file):
                logger.warning(f"Transformer file not found: {transformer_file}")
                return False
            
            with open(transformer_file, 'rb') as f:
                transformers = pickle.load(f)
            
            self.scaler = transformers['scaler']
            self.encoder = transformers['encoder']
            self.feature_columns = transformers['feature_columns']
            self.target_columns = transformers['target_columns']
            
            logger.info(f"Transformers loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformers: {e}")
            return False


def main():
    """
    Main function to test the feature engineer.
    """
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'video_id': [f'video_{i}' for i in range(n_samples)],
            'title': [f'How to cook pasta {i}' for i in range(n_samples)],
            'description': [f'This video shows how to cook delicious pasta dish {i}' for i in range(n_samples)],
            'tags': [f'cooking|pasta|recipe|food' for i in range(n_samples)],
            'category_id': np.random.choice([26, 20, 10, 22], n_samples),  # Howto, Gaming, Music, People
            'channel_title': [f'Cooking Channel {i%10}' for i in range(n_samples)],
            'publish_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'trending_date': pd.date_range('2023-01-02', periods=n_samples, freq='H'),
            'region_code': np.random.choice(['US', 'GB', 'CA', 'DE'], n_samples),
            'views': np.random.randint(1000, 1000000, n_samples),
            'likes': np.random.randint(10, 10000, n_samples),
            'comments': np.random.randint(1, 1000, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Initialize feature engineer
        engineer = YouTubeFeatureEngineer()
        
        # Prepare features
        X, y = engineer.prepare_features_for_ml(df)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target matrix shape: {y.shape}")
        print(f"Feature columns: {len(engineer.feature_columns)}")
        
        # Get feature importance data
        importance_data = engineer.get_feature_importance_data(df)
        
        print(f"\nFeature breakdown:")
        print(f"Total features: {importance_data['feature_count']}")
        print(f"Numeric features: {len(importance_data['numeric_features'])}")
        print(f"Categorical features: {len(importance_data['categorical_features'])}")
        print(f"Interaction features: {len(importance_data['interaction_features'])}")
        
        # Save transformers
        engineer.save_transformers('models/feature_engineering')
        
        print("\nFeature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
