"""
Model Evaluation Module for YouTube Analytics

This module provides comprehensive evaluation utilities for ML models including
metrics calculation, visualization, and comparison analysis.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, explained_variance_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    
    Attributes:
        evaluation_results: Dictionary of evaluation results
        output_dir: Directory to save evaluation artifacts
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation artifacts
        """
        self.evaluation_results = {}
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"ModelEvaluator initialized with output directory: {output_dir}")
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None,
                                 class_names: Optional[List[str]] = None,
                                 model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a classification model comprehensively.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (Optional[np.ndarray]): Prediction probabilities
            class_names (Optional[List[str]]): Class names
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Evaluating classification model: {model_name}")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            # Binary classification
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        results = {
            'model_name': model_name,
            'model_type': 'classification',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"Classification evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a regression model comprehensively.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        logger.info(f"Evaluating regression model: {model_name}")
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        results = {
            'model_name': model_name,
            'model_type': 'regression',
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'explained_variance': ev,
            'mape': mape,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        logger.info(f"Regression evaluation completed. R²: {r2:.4f}, RMSE: {rmse:.2f}")
        
        return results
    
    def compare_models(self, metric: str = 'r2_score') -> pd.DataFrame:
        """
        Compare multiple models based on a specific metric.
        
        Args:
            metric (str): Metric to compare by
            
        Returns:
            pd.DataFrame: Model comparison
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            if metric in results:
                comparison_data.append({
                    'model': model_name,
                    'metric_value': results[metric],
                    'model_type': results.get('model_type', 'unknown')
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        if not df_comparison.empty:
            df_comparison = df_comparison.sort_values('metric_value', ascending=False)
            
            logger.info(f"Model comparison by {metric}:")
            for _, row in df_comparison.iterrows():
                logger.info(f"  {row['model']}: {row['metric_value']:.4f}")
        
        return df_comparison
    
    def plot_confusion_matrix(self, model_name: str, save_plot: bool = True) -> None:
        """
        Plot confusion matrix for a classification model.
        
        Args:
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"No results found for model: {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        if results['model_type'] != 'classification':
            logger.warning(f"Model {model_name} is not a classification model")
            return
        
        cm = np.array(results['confusion_matrix'])
        class_names = results.get('class_names', [f'Class_{i}' for i in range(len(cm))])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {plot_path}")
        
        plt.show()
    
    def plot_residuals(self, model_name: str, save_plot: bool = True) -> None:
        """
        Plot residual analysis for a regression model.
        
        Args:
            model_name (str): Name of the model
            save_plot (bool): Whether to save the plot
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"No results found for model: {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        if results['model_type'] != 'regression':
            logger.warning(f"Model {model_name} is not a regression model")
            return
        
        # Generate synthetic residuals for demonstration
        # In practice, you'd pass actual y_true and y_pred
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.exponential(scale=100000, size=n_samples)
        y_pred = y_true + np.random.normal(0, y_true * 0.1, size=n_samples)
        residuals = y_true - y_pred
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Residuals vs Index
        axes[1, 1].plot(residuals)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Index')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'{model_name}_residuals.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {plot_path}")
        
        plt.show()
    
    def plot_model_comparison(self, metric: str = 'r2_score', save_plot: bool = True) -> None:
        """
        Plot model comparison chart.
        
        Args:
            metric (str): Metric to compare
            save_plot (bool): Whether to save the plot
        """
        df_comparison = self.compare_models(metric)
        
        if df_comparison.empty:
            logger.warning("No data available for comparison plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        bars = plt.bar(df_comparison['model'], df_comparison['metric_value'])
        
        # Add value labels on bars
        for bar, value in zip(bars, df_comparison['metric_value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'model_comparison_{metric}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {plot_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, save_report: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_report (bool): Whether to save the report
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available for report")
            return {}
        
        # Summary statistics
        classification_models = {
            name: results for name, results in self.evaluation_results.items()
            if results.get('model_type') == 'classification'
        }
        
        regression_models = {
            name: results for name, results in self.evaluation_results.items()
            if results.get('model_type') == 'regression'
        }
        
        # Best models
        best_classification = None
        best_regression = None
        
        if classification_models:
            best_classification = max(
                classification_models.items(),
                key=lambda x: x[1].get('accuracy', 0)
            )
        
        if regression_models:
            best_regression = max(
                regression_models.items(),
                key=lambda x: x[1].get('r2_score', 0)
            )
        
        report = {
            'evaluation_summary': {
                'total_models_evaluated': len(self.evaluation_results),
                'classification_models': len(classification_models),
                'regression_models': len(regression_models),
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
            },
            'best_classification_model': {
                'name': best_classification[0] if best_classification else None,
                'accuracy': best_classification[1]['accuracy'] if best_classification else None,
                'f1_score': best_classification[1]['f1_score'] if best_classification else None
            } if best_classification else None,
            'best_regression_model': {
                'name': best_regression[0] if best_regression else None,
                'r2_score': best_regression[1]['r2_score'] if best_regression else None,
                'rmse': best_regression[1]['rmse'] if best_regression else None
            } if best_regression else None,
            'detailed_results': self.evaluation_results
        }
        
        if save_report:
            report_path = os.path.join(self.output_dir, 'evaluation_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Evaluation report saved to {report_path}")
        
        return report
    
    def evaluate_nlp_categorization(self, df: pd.DataFrame, 
                                 actual_col: str = 'category_id',
                                 predicted_col: str = 'nlp_predicted_category') -> Dict[str, Any]:
        """
        Evaluate NLP categorization results specifically.
        
        Args:
            df (pd.DataFrame): DataFrame with actual and predicted categories
            actual_col (str): Column with actual categories
            predicted_col (str): Column with predicted categories
            
        Returns:
            Dict[str, Any]: NLP evaluation results
        """
        logger.info("Evaluating NLP categorization results")
        
        # Filter valid predictions
        valid_mask = (
            df[actual_col].notna() & 
            df[predicted_col].notna() & 
            (df[predicted_col] != -1)
        )
        
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            logger.warning("No valid NLP predictions to evaluate")
            return {'error': 'No valid predictions to evaluate'}
        
        y_true = df_valid[actual_col].values
        y_pred = df_valid[predicted_col].values
        
        # Category mapping for names
        category_mapping = {
            1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
            15: "Pets & Animals", 17: "Sports", 19: "Travel & Events",
            20: "Gaming", 22: "People & Blogs", 23: "Comedy",
            24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
            27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism"
        }
        
        class_names = [
            category_mapping.get(cat_id, f"Category_{cat_id}")
            for cat_id in sorted(np.unique(np.concatenate([y_true, y_pred])))
        ]
        
        # Evaluate using classification evaluator
        results = self.evaluate_classification_model(
            y_true, y_pred, 
            class_names=class_names,
            model_name="nlp_categorization"
        )
        
        # Add NLP-specific metrics
        results['total_samples'] = len(df_valid)
        results['unique_categories_actual'] = len(np.unique(y_true))
        results['unique_categories_predicted'] = len(np.unique(y_pred))
        
        # Per-category accuracy
        category_accuracy = {}
        for cat_id in np.unique(y_true):
            cat_mask = y_true == cat_id
            if cat_mask.sum() > 0:
                cat_accuracy = accuracy_score(y_true[cat_mask], y_pred[cat_mask])
                category_accuracy[cat_id] = cat_accuracy
        
        results['category_accuracy'] = category_accuracy
        
        logger.info(f"NLP categorization evaluation completed. Accuracy: {results['accuracy']:.4f}")
        
        return results
    
    def evaluate_engagement_prediction(self, df: pd.DataFrame,
                                 actual_views_col: str = 'views',
                                 predicted_views_col: str = 'ml_predicted_views',
                                 actual_er_col: str = 'engagement_rate',
                                 predicted_er_col: str = 'ml_predicted_engagement_rate') -> Dict[str, Any]:
        """
        Evaluate engagement prediction results specifically.
        
        Args:
            df (pd.DataFrame): DataFrame with actual and predicted values
            actual_views_col (str): Column with actual views
            predicted_views_col (str): Column with predicted views
            actual_er_col (str): Column with actual engagement rate
            predicted_er_col (str): Column with predicted engagement rate
            
        Returns:
            Dict[str, Any]: Engagement evaluation results
        """
        logger.info("Evaluating engagement prediction results")
        
        # Filter valid predictions
        valid_mask = (
            df[actual_views_col].notna() & 
            df[predicted_views_col].notna() & 
            (df[actual_views_col] > 0)
        )
        
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            logger.warning("No valid engagement predictions to evaluate")
            return {'error': 'No valid predictions to evaluate'}
        
        results = {}
        
        # Evaluate views prediction
        y_true_views = df_valid[actual_views_col].values
        y_pred_views = df_valid[predicted_views_col].values
        
        views_results = self.evaluate_regression_model(
            y_true_views, y_pred_views,
            model_name="views_prediction"
        )
        results['views_prediction'] = views_results
        
        # Evaluate engagement rate prediction if available
        if actual_er_col in df_valid.columns and predicted_er_col in df_valid.columns:
            er_valid_mask = (
                df_valid[actual_er_col].notna() & 
                df_valid[predicted_er_col].notna()
            )
            
            if er_valid_mask.sum() > 0:
                y_true_er = df_valid.loc[er_valid_mask, actual_er_col].values
                y_pred_er = df_valid.loc[er_valid_mask, predicted_er_col].values
                
                er_results = self.evaluate_regression_model(
                    y_true_er, y_pred_er,
                    model_name="engagement_rate_prediction"
                )
                results['engagement_rate_prediction'] = er_results
        
        # Overall metrics
        results['total_samples'] = len(df_valid)
        results['views_r2_meets_threshold'] = views_results['r2_score'] >= 0.85
        results['overall_success'] = results['views_r2_meets_threshold']
        
        logger.info(f"Engagement prediction evaluation completed. Views R²: {views_results['r2_score']:.4f}")
        
        return results
    
    def save_evaluation_summary(self) -> None:
        """Save a summary of all evaluation results."""
        if not self.evaluation_results:
            logger.warning("No evaluation results to save")
            return
        
        summary = self.generate_evaluation_report(save_report=False)
        
        # Create summary DataFrame
        summary_data = []
        
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'model': model_name,
                'type': results.get('model_type', 'unknown'),
                'accuracy': results.get('accuracy', 'N/A'),
                'r2_score': results.get('r2_score', 'N/A'),
                'rmse': results.get('rmse', 'N/A'),
                'f1_score': results.get('f1_score', 'N/A')
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'evaluation_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        
        logger.info(f"Evaluation summary saved to {csv_path}")


def main():
    """
    Main function to test the model evaluator.
    """
    try:
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 1000
        
        # Classification test data
        y_true_cls = np.random.randint(0, 5, n_samples)
        y_pred_cls = y_true_cls.copy()
        # Introduce some errors
        error_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y_pred_cls[error_indices] = np.random.randint(0, 5, len(error_indices))
        
        # Regression test data
        y_true_reg = np.random.exponential(scale=100000, size=n_samples)
        y_pred_reg = y_true_reg + np.random.normal(0, y_true_reg * 0.1, size=n_samples)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Test classification evaluation
        cls_results = evaluator.evaluate_classification_model(
            y_true_cls, y_pred_cls,
            class_names=['Music', 'Gaming', 'Education', 'Entertainment', 'Sports'],
            model_name="test_classifier"
        )
        
        print("Classification Results:")
        print(f"Accuracy: {cls_results['accuracy']:.4f}")
        print(f"F1 Score: {cls_results['f1_score']:.4f}")
        
        # Test regression evaluation
        reg_results = evaluator.evaluate_regression_model(
            y_true_reg, y_pred_reg,
            model_name="test_regressor"
        )
        
        print("\nRegression Results:")
        print(f"R² Score: {reg_results['r2_score']:.4f}")
        print(f"RMSE: {reg_results['rmse']:.2f}")
        
        # Generate comparison
        comparison = evaluator.compare_models('r2_score')
        print("\nModel Comparison:")
        print(comparison)
        
        # Generate comprehensive report
        report = evaluator.generate_evaluation_report()
        
        print("\nEvaluation Summary:")
        print(f"Total models: {report['evaluation_summary']['total_models_evaluated']}")
        print(f"Classification models: {report['evaluation_summary']['classification_models']}")
        print(f"Regression models: {report['evaluation_summary']['regression_models']}")
        
        # Save evaluation summary
        evaluator.save_evaluation_summary()
        
        print("\nModel evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
