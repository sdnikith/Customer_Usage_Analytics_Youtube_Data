"""
ML Engagement Prediction Module

This module trains Random Forest and Gradient Boosting models to predict
video engagement metrics (views and engagement_rate) with target R² > 0.85.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EngagementPredictor:
    """Trains Random Forest and Gradient Boosting regressors to predict views and engagement_rate."""

    def __init__(self, model_path: str = "models/engagement_predictor"):
        self.model_path = model_path
        self.rf_model = None
        self.gb_model = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_variables = ['views', 'engagement_rate']
        self.model_performance = {}
        self.r2_threshold = float(os.getenv('ML_R2_THRESHOLD', '0.85'))
        
        logger.info(f"EngagementPredictor initialized with R² threshold: {self.r2_threshold}")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Selects numeric features, fills missing values, and log-transforms views."""
        logger.info("Preparing data for engagement prediction")
        
        # Make a copy to avoid modifying original
        df_prep = df.copy()
        
        # Define feature columns (exclude targets and identifiers)
        exclude_columns = [
            'video_id', 'title', 'description', 'tags', 'channel_title',
            'channel_id', 'publish_date', 'trending_date', 'extracted_at',
            'category_name', 'thumbnail_url', 'nlp_predicted_category',
            'nlp_category_confidence', 'ml_predicted_views', 
            'ml_predicted_engagement_rate', 'processed_at'
        ]
        
        # Add target variables to exclude
        exclude_columns.extend(self.target_variables)
        
        # Get feature columns
        feature_columns = [col for col in df_prep.columns if col not in exclude_columns]
        
        # Filter to only numeric features for regression
        numeric_features = df_prep[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Selected {len(numeric_features)} numeric features")
        
        # Handle missing values
        df_features = df_prep[numeric_features].copy()
        df_features = df_features.fillna(0)
        
        # Prepare targets
        df_targets = df_prep[self.target_variables].copy()
        
        # Handle missing targets
        for target in self.target_variables:
            if target in df_targets.columns:
                df_targets[target] = pd.to_numeric(df_targets[target], errors='coerce').fillna(0)
        
        # Remove rows with zero views (can't predict log(views) properly)
        valid_mask = df_targets['views'] > 0
        df_features = df_features[valid_mask]
        df_targets = df_targets[valid_mask]
        
        # Log transform views for better model performance
        df_targets['log_views'] = np.log1p(df_targets['views'])
        
        logger.info(f"Final dataset: {df_features.shape[0]} samples, {df_features.shape[1]} features")
        
        self.feature_names = df_features.columns.tolist()
        
        return df_features, df_targets
    
    def _fit_regressor(self, X: pd.DataFrame, y: pd.Series,
                       target_name: str, model) -> Dict[str, Any]:
        """Fit any sklearn regressor and return a standardised results dict."""
        model_type = type(model).__name__
        logger.info(f"Training {model_type} for {target_name}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)

        train_metrics = self._calculate_metrics(y_train, model.predict(X_train_scaled), target_name, "train")
        test_metrics = self._calculate_metrics(y_test, model.predict(X_test_scaled), target_name, "test")

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)

        feature_importance = (
            pd.DataFrame({'feature': self.feature_names, 'importance': model.feature_importances_})
            .sort_values('importance', ascending=False)
            .head(20)
            .to_dict('records')
        )

        logger.info(f"{model_type} trained for {target_name}. Test R²: {test_metrics['r2']:.4f}")
        return {
            'model': model,
            'model_type': model_type,
            'target': target_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean_r2': float(cv_scores.mean()),
            'cv_std_r2': float(cv_scores.std()),
            'feature_importance': feature_importance,
            'meets_threshold': test_metrics['r2'] >= self.r2_threshold,
            'training_timestamp': datetime.now(timezone.utc).isoformat(),
        }

    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1,
        )
        return self._fit_regressor(X, y, target_name, model)

    def train_gradient_boosting(self, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42,
        )
        return self._fit_regressor(X, y, target_name, model)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         target_name: str, dataset: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            target_name (str): Name of target
            dataset (str): Dataset type (train/test)
            
        Returns:
            Dict[str, float]: Metrics dictionary
        """
        # Handle log-transformed views
        if 'log_views' in target_name:
            y_true_original = np.expm1(y_true)
            y_pred_original = np.expm1(y_pred)
        else:
            y_true_original = y_true
            y_pred_original = y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        
        # Calculate MAPE (avoid division by zero)
        mask = y_true_original != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(
                y_true_original[mask], y_pred_original[mask]
            )
        else:
            mape = np.inf
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': ev,
            'mape': mape if mape != np.inf else 0,
            'target': target_name,
            'dataset': dataset
        }
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both Random Forest and Gradient Boosting models.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            Dict[str, Any]: Training results for all models
        """
        logger.info("Starting model training for engagement prediction")
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        all_results = {}
        best_models = {}
        
        # Train models for each target
        for target in self.target_variables:
            if target not in y.columns:
                logger.warning(f"Target {target} not found in data")
                continue
            
            y_target = y[target]
            
            # Train Random Forest
            rf_results = self.train_random_forest(X, y_target, target)
            all_results[f'{target}_random_forest'] = rf_results
            
            # Train Gradient Boosting
            gb_results = self.train_gradient_boosting(X, y_target, target)
            all_results[f'{target}_gradient_boosting'] = gb_results
            
            # Select best model for this target
            rf_r2 = rf_results['test_metrics']['r2']
            gb_r2 = gb_results['test_metrics']['r2']
            
            if gb_r2 > rf_r2:
                best_models[target] = gb_results
                logger.info(f"Best model for {target}: Gradient Boosting (R²: {gb_r2:.4f})")
            else:
                best_models[target] = rf_results
                logger.info(f"Best model for {target}: Random Forest (R²: {rf_r2:.4f})")
        
        # Store results
        self.model_performance = all_results
        
        # Store best models
        self.rf_model = best_models.get('views', {}).get('model')
        self.gb_model = best_models.get('engagement_rate', {}).get('model')
        
        # Select overall best model (prioritize views prediction)
        if 'views' in best_models:
            self.best_model = best_models['views']['model']
        elif best_models:
            self.best_model = list(best_models.values())[0]['model']
        
        # Prepare summary
        summary = {
            'total_models_trained': len(all_results),
            'targets_trained': list(best_models.keys()),
            'best_models': {
                target: {
                    'model_type': results['model_type'],
                    'r2_score': results['test_metrics']['r2'],
                    'meets_threshold': results['meets_threshold']
                }
                for target, results in best_models.items()
            },
            'overall_performance': self._get_overall_performance(all_results),
            'training_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Model training completed. Trained {len(all_results)} models")
        
        return summary
    
    def _get_overall_performance(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall performance summary.
        
        Args:
            all_results (Dict[str, Any]): All model results
            
        Returns:
            Dict[str, Any]: Performance summary
        """
        r2_scores = []
        models_meeting_threshold = 0
        
        for model_key, results in all_results.items():
            r2 = results['test_metrics']['r2']
            r2_scores.append(r2)
            
            if results['meets_threshold']:
                models_meeting_threshold += 1
        
        return {
            'average_r2': np.mean(r2_scores),
            'max_r2': np.max(r2_scores),
            'min_r2': np.min(r2_scores),
            'models_meeting_threshold': models_meeting_threshold,
            'total_models': len(all_results),
            'threshold_met_percentage': (models_meeting_threshold / len(all_results)) * 100
        }
    
    def predict_engagement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make engagement predictions on new data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with predictions
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Call train_models() first.")
        
        logger.info(f"Making engagement predictions for {len(df)} records")
        
        # Prepare features (same as training)
        X, _ = self.prepare_data(df)
        
        if len(X) == 0:
            logger.warning("No valid data for prediction")
            df['ml_predicted_views'] = None
            df['ml_predicted_engagement_rate'] = None
            return df
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.best_model.predict(X_scaled)
        
        # Handle log-transformed predictions
        if 'log_views' in str(self.model_performance):
            predicted_views = np.expm1(predictions)
        else:
            predicted_views = predictions
        
        # Add predictions to DataFrame
        df_result = df.copy()
        
        # Align predictions with original DataFrame
        if len(X) == len(df_result):
            df_result['ml_predicted_views'] = predicted_views
        else:
            # Handle case where some rows were filtered out
            df_result.loc[X.index, 'ml_predicted_views'] = predicted_views
        
        # Predict engagement rate if we have a model for it
        if 'engagement_rate' in self.model_performance:
            # Find the engagement rate model
            er_model_key = None
            for key, results in self.model_performance.items():
                if 'engagement_rate' in key and results['meets_threshold']:
                    er_model_key = key
                    break
            
            if er_model_key:
                er_model = self.model_performance[er_model_key]['model']
                er_predictions = er_model.predict(X_scaled)
                
                if len(X) == len(df_result):
                    df_result['ml_predicted_engagement_rate'] = er_predictions
                else:
                    df_result.loc[X.index, 'ml_predicted_engagement_rate'] = er_predictions
        
        # Ensure non-negative predictions
        df_result['ml_predicted_views'] = df_result['ml_predicted_views'].clip(lower=0)
        if 'ml_predicted_engagement_rate' in df_result.columns:
            df_result['ml_predicted_engagement_rate'] = df_result['ml_predicted_engagement_rate'].clip(lower=0)
        
        logger.info("Engagement predictions completed")
        
        return df_result
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate predictions against actual values.
        
        Args:
            df (pd.DataFrame): Data with predictions and actual values
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        required_columns = ['ml_predicted_views', 'views']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain {required_columns}")
        
        # Filter to valid predictions
        valid_mask = (
            df['ml_predicted_views'].notna() & 
            df['views'].notna() & 
            (df['views'] > 0)
        )
        
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            logger.warning("No valid predictions to evaluate")
            return {'error': 'No valid predictions to evaluate'}
        
        # Calculate metrics for views
        y_true_views = df_valid['views'].values
        y_pred_views = df_valid['ml_predicted_views'].values
        
        views_metrics = {
            'mse': mean_squared_error(y_true_views, y_pred_views),
            'rmse': np.sqrt(mean_squared_error(y_true_views, y_pred_views)),
            'mae': mean_absolute_error(y_true_views, y_pred_views),
            'r2': r2_score(y_true_views, y_pred_views),
            'mape': mean_absolute_percentage_error(y_true_views, y_pred_views)
        }
        
        # Calculate metrics for engagement rate if available
        er_metrics = None
        if 'ml_predicted_engagement_rate' in df_valid.columns and 'engagement_rate' in df_valid.columns:
            y_true_er = df_valid['engagement_rate'].values
            y_pred_er = df_valid['ml_predicted_engagement_rate'].values
            
            er_valid_mask = (
                df_valid['ml_predicted_engagement_rate'].notna() & 
                df_valid['engagement_rate'].notna()
            )
            
            if er_valid_mask.sum() > 0:
                er_metrics = {
                    'mse': mean_squared_error(y_true_er[er_valid_mask], y_pred_er[er_valid_mask]),
                    'rmse': np.sqrt(mean_squared_error(y_true_er[er_valid_mask], y_pred_er[er_valid_mask])),
                    'mae': mean_absolute_error(y_true_er[er_valid_mask], y_pred_er[er_valid_mask]),
                    'r2': r2_score(y_true_er[er_valid_mask], y_pred_er[er_valid_mask])
                }
        
        results = {
            'total_samples': len(df_valid),
            'views_metrics': views_metrics,
            'engagement_rate_metrics': er_metrics,
            'views_r2_meets_threshold': views_metrics['r2'] >= self.r2_threshold,
            'overall_success': views_metrics['r2'] >= self.r2_threshold,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Evaluation completed. Views R²: {views_metrics['r2']:.4f}")
        
        return results
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """
        Get feature importance summary from trained models.
        
        Returns:
            Dict[str, Any]: Feature importance summary
        """
        if not self.model_performance:
            return {'error': 'No trained models available'}
        
        importance_data = {}
        
        for model_key, results in self.model_performance.items():
            if 'feature_importance' in results:
                target = results['target']
                model_type = results['model_type']
                
                if target not in importance_data:
                    importance_data[target] = {}
                
                importance_data[target][model_type] = results['feature_importance']
        
        return {
            'feature_importance_by_target': importance_data,
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'summary_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def save_models(self) -> bool:
        """
        Save trained models to disk.
        
        Returns:
            bool: True if saved successfully
        """
        if not self.model_performance:
            logger.error("No trained models to save")
            return False
        
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save models and metadata
            model_data = {
                'best_model': self.best_model,
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'target_variables': self.target_variables,
                'model_performance': self.model_performance,
                'r2_threshold': self.r2_threshold,
                'save_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            model_file = os.path.join(self.model_path, 'engagement_predictor.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Models saved to {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            bool: True if loaded successfully
        """
        model_file = os.path.join(self.model_path, 'engagement_predictor.pkl')
        
        if not os.path.exists(model_file):
            logger.warning(f"Model file not found: {model_file}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['best_model']
            self.rf_model = model_data['rf_model']
            self.gb_model = model_data['gb_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.target_variables = model_data['target_variables']
            self.model_performance = model_data['model_performance']
            self.r2_threshold = model_data['r2_threshold']
            
            logger.info(f"Models loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


def main():
    """
    Main function to test the engagement predictor.
    """
    try:
        n_samples = 1000
        rng = np.random.default_rng(42)
        title_templates = [
            "Python tutorial for beginners",
            "Top 10 gaming moments",
            "Easy pasta recipe",
            "Travel vlog Tokyo",
            "Stand-up comedy special",
            "Machine learning explained",
            "Guitar lesson fingerpicking",
            "Morning workout routine",
            "Book review thriller",
            "DIY home improvement",
        ]
        data = {
            'video_id': [f'vid_{i:04d}' for i in range(n_samples)],
            'title': [title_templates[i % len(title_templates)] for i in range(n_samples)],
            'description': [f'Full walkthrough — part {i + 1}' for i in range(n_samples)],
            'category_id': rng.integers(1, 30, n_samples),
            'channel_title': [f'Channel {i % 10}' for i in range(n_samples)],
            'publish_date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'trending_date': pd.date_range('2023-01-02', periods=n_samples, freq='H'),
            'region_code': rng.choice(['US', 'GB', 'CA', 'DE'], n_samples),
            'views': rng.integers(1000, 1_000_000, n_samples),
            'likes': rng.integers(10, 10_000, n_samples),
            'comments': rng.integers(1, 1_000, n_samples),
            'title_length': rng.integers(10, 100, n_samples),
            'description_length': rng.integers(50, 500, n_samples),
            'tag_count': rng.integers(0, 20, n_samples),
            'engagement_rate': rng.uniform(0.001, 0.1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Initialize predictor
        predictor = EngagementPredictor()
        
        # Train models
        results = predictor.train_models(df)
        
        print("Training Results:")
        print(f"Total models trained: {results['total_models_trained']}")
        print(f"Targets trained: {results['targets_trained']}")
        print(f"Overall performance: {results['overall_performance']}")
        
        # Make predictions
        df_with_predictions = predictor.predict_engagement(df)
        
        # Evaluate predictions
        evaluation = predictor.evaluate_predictions(df_with_predictions)
        
        print("\nEvaluation Results:")
        if 'views_metrics' in evaluation:
            views_metrics = evaluation['views_metrics']
            print(f"Views R²: {views_metrics['r2']:.4f}")
            print(f"Views RMSE: {views_metrics['rmse']:.2f}")
            print(f"Views MAE: {views_metrics['mae']:.2f}")
        
        # Get feature importance
        importance = predictor.get_feature_importance_summary()
        print(f"\nFeature importance summary created")
        
        # Save models
        predictor.save_models()
        
        print("\nEngagement prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
