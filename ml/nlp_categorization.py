"""
NLP Video Categorization Module

This module uses NLTK/spaCy for text preprocessing, TF-IDF vectorization,
and machine learning classifiers to predict YouTube video categories from text content.
Target: 85%+ classification accuracy.
"""

import os
import re
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class NLPVideoCategorizer:
    """
    NLP-based video categorizer using text features.
    
    Attributes:
        vectorizer: TF-IDF vectorizer
        classifier: Trained classifier
        label_encoder: Label encoder for categories
        category_mapping: Mapping from category_id to category_name
        accuracy_threshold: Minimum accuracy threshold
        model_path: Path to save/load models
    """
    
    def __init__(self, model_path: str = "models/nlp_categorizer", use_spacy: bool = True):
        """
        Initialize the NLP categorizer.
        
        Args:
            model_path (str): Path to save/load models
            use_spacy (bool): Whether to use spaCy or NLTK for preprocessing
        """
        self.model_path = model_path
        self.use_spacy = use_spacy
        self.accuracy_threshold = float(os.getenv('NLP_ACCURACY_THRESHOLD', '0.85'))
        
        # Initialize components
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        
        # Initialize NLP components
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found, falling back to NLTK")
                self.use_spacy = False
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
        else:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        
        # YouTube category mapping
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
        
        logger.info(f"NLPVideoCategorizer initialized with spaCy={use_spacy}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, tokenizing, and lemmatizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.use_spacy:
            # Use spaCy for processing
            doc = self.nlp(text)
            tokens = [
                token.lemma_ for token in doc 
                if not token.is_stop and not token.is_punct and len(token) > 2
            ]
        else:
            # Use NLTK for processing
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token) for token in tokens
                if token not in self.stop_words and token.isalpha() and len(token) > 2
            ]
        
        return ' '.join(tokens)
    
    def combine_text_features(self, row: pd.Series) -> str:
        """
        Combine title, description, and tags into a single text feature.
        
        Args:
            row (pd.Series): Row containing text fields
            
        Returns:
            str: Combined text
        """
        title = row.get('title', '')
        description = row.get('description', '')
        tags = row.get('tags', '')
        
        # Handle different tag formats
        if isinstance(tags, str):
            # Split by common delimiters
            tags = re.sub(r'[|,;]', ' ', tags)
        
        # Combine with weights
        combined = f"{title} {title} "  # Title gets more weight
        combined += f"{description} "  # Description
        combined += f"{tags}"  # Tags
        
        return self.preprocess_text(combined)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and labels
        """
        logger.info("Preparing features for NLP categorization")
        
        # Filter to only include categories we have in mapping
        valid_categories = list(self.category_mapping.keys())
        df_filtered = df[df['category_id'].isin(valid_categories)].copy()
        
        logger.info(f"Filtered to {len(df_filtered)} records with valid categories")
        
        # Combine text features
        df_filtered['combined_text'] = df_filtered.apply(
            self.combine_text_features, axis=1
        )
        
        # Remove empty text
        df_filtered = df_filtered[df_filtered['combined_text'].str.len() > 10]
        
        logger.info(f"After filtering empty text: {len(df_filtered)} records")
        
        # Prepare features and labels
        X = df_filtered['combined_text'].values
        y = df_filtered['category_id'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Features prepared: {X.shape[0]} samples, {len(np.unique(y_encoded))} categories")
        
        return X, y_encoded
    
    def create_pipeline(self, classifier_type: str = 'naive_bayes') -> Pipeline:
        """
        Create a scikit-learn pipeline with vectorizer and classifier.
        
        Args:
            classifier_type (str): Type of classifier ('naive_bayes', 'svm', 'random_forest')
            
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        # Choose classifier
        if classifier_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=0.1)
        elif classifier_type == 'svm':
            classifier = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline
    
    def train_model(self, df: pd.DataFrame, classifier_type: str = 'naive_bayes') -> Dict[str, Any]:
        """
        Train the NLP categorization model.
        
        Args:
            df (pd.DataFrame): Training data
            classifier_type (str): Type of classifier to use
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training NLP categorization model with {classifier_type}")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train pipeline
        self.classifier = self.create_pipeline(classifier_type)
        
        # Train with cross-validation
        cv_scores = cross_val_score(
            self.classifier, X_train, y_train, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Fit on full training set
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_names = [
            self.category_mapping.get(cat_id, f"Category_{cat_id}")
            for cat_id in self.label_encoder.inverse_transform(
                np.unique(np.concatenate([y_test, y_pred]))
            )
        ]
        
        report = classification_report(
            y_test, y_pred, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Prepare results
        results = {
            'classifier_type': classifier_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self._get_feature_importance(),
            'meets_threshold': accuracy >= self.accuracy_threshold,
            'training_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Training completed. Test accuracy: {accuracy:.4f} (threshold: {self.accuracy_threshold})")
        
        return results
    
    def _get_feature_importance(self) -> Optional[List[Tuple[str, float]]]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Optional[List[Tuple[str, float]]]: Feature importance scores
        """
        if self.classifier is None:
            return None
        
        # Get feature names from vectorizer
        feature_names = self.classifier.named_steps['vectorizer'].get_feature_names_out()
        
        # Get importance based on classifier type
        classifier = self.classifier.named_steps['classifier']
        
        if hasattr(classifier, 'coef_'):
            # Linear models (Naive Bayes, SVM)
            importance = np.abs(classifier.coef_).mean(axis=0)
        elif hasattr(classifier, 'feature_importances_'):
            # Tree-based models (Random Forest)
            importance = classifier.feature_importances_
        else:
            return None
        
        # Create feature importance list
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:20]  # Top 20 features
    
    def predict_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict categories for new data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with predictions
        """
        if self.classifier is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        logger.info(f"Predicting categories for {len(df)} records")
        
        # Prepare features
        df_copy = df.copy()
        df_copy['combined_text'] = df_copy.apply(self.combine_text_features, axis=1)
        
        # Filter out empty text
        mask = df_copy['combined_text'].str.len() > 10
        df_predict = df_copy[mask].copy()
        
        if len(df_predict) == 0:
            logger.warning("No valid text for prediction")
            df['nlp_predicted_category'] = None
            df['nlp_category_confidence'] = None
            return df
        
        # Make predictions
        X = df_predict['combined_text'].values
        y_pred = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Convert predictions back to original category IDs
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Get confidence scores
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        # Add predictions to DataFrame
        df.loc[mask, 'nlp_predicted_category'] = y_pred_original
        df.loc[mask, 'nlp_category_confidence'] = confidence_scores
        
        # Fill missing predictions
        df['nlp_predicted_category'] = df['nlp_predicted_category'].fillna(-1).astype(int)
        df['nlp_category_confidence'] = df['nlp_category_confidence'].fillna(0.0)
        
        logger.info(f"Predictions completed for {len(df_predict)} records")
        
        return df
    
    def evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate predictions against actual categories.
        
        Args:
            df (pd.DataFrame): Data with predictions and actual categories
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if 'nlp_predicted_category' not in df.columns or 'category_id' not in df.columns:
            raise ValueError("DataFrame must contain both predictions and actual categories")
        
        # Filter out records with no predictions
        valid_mask = (
            df['nlp_predicted_category'] != -1 & 
            df['category_id'].isin(self.category_mapping.keys())
        )
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            logger.warning("No valid predictions to evaluate")
            return {'error': 'No valid predictions to evaluate'}
        
        y_true = df_valid['category_id'].values
        y_pred = df_valid['nlp_predicted_category'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Generate detailed report
        class_names = [
            self.category_mapping.get(cat_id, f"Category_{cat_id}")
            for cat_id in sorted(np.unique(np.concatenate([y_true, y_pred])))
        ]
        
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Calculate accuracy by category
        category_accuracy = {}
        for cat_id in np.unique(y_true):
            cat_mask = y_true == cat_id
            if cat_mask.sum() > 0:
                cat_accuracy = accuracy_score(y_true[cat_mask], y_pred[cat_mask])
                category_accuracy[cat_id] = cat_accuracy
        
        results = {
            'total_samples': len(df_valid),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'meets_threshold': accuracy >= self.accuracy_threshold,
            'classification_report': report,
            'category_accuracy': category_accuracy,
            'evaluation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        
        return results
    
    def save_model(self) -> bool:
        """
        Save the trained model to disk.
        
        Returns:
            bool: True if saved successfully
        """
        if self.classifier is None:
            logger.error("No model to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save model components
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'category_mapping': self.category_mapping,
                'use_spacy': self.use_spacy,
                'accuracy_threshold': self.accuracy_threshold,
                'training_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            model_file = os.path.join(self.model_path, 'nlp_categorizer.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            bool: True if loaded successfully
        """
        model_file = os.path.join(self.model_path, 'nlp_categorizer.pkl')
        
        if not os.path.exists(model_file):
            logger.warning(f"Model file not found: {model_file}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.category_mapping = model_data['category_mapping']
            self.use_spacy = model_data['use_spacy']
            self.accuracy_threshold = model_data['accuracy_threshold']
            
            logger.info(f"Model loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_predictions_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> bool:
        """
        Save predictions to S3.
        
        Args:
            df (pd.DataFrame): DataFrame with predictions
            bucket (str): S3 bucket name
            key (str): S3 key
            
        Returns:
            bool: True if saved successfully
        """
        try:
            import boto3
            
            s3_client = boto3.client('s3')
            
            # Save as parquet
            buffer = df.to_parquet(index=False)
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buffer,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Predictions saved to s3://{bucket}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving predictions to S3: {e}")
            return False


def main():
    """
    Main function to test the NLP categorizer.
    """
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'video_id': [f'video_{i}' for i in range(n_samples)],
            'title': [f'How to cook pasta {i}' if i % 3 == 0 else f'Gaming highlights {i}' for i in range(n_samples)],
            'description': [f'This video shows how to cook delicious pasta dish {i}' if i % 3 == 0 else f'Amazing gaming moments and highlights {i}' for i in range(n_samples)],
            'tags': [f'cooking|pasta|recipe|food' if i % 3 == 0 else f'gaming|highlights|gameplay' for i in range(n_samples)],
            'category_id': [26 if i % 3 == 0 else 20 for i in range(n_samples)]  # 26=Howto, 20=Gaming
        }
        
        df = pd.DataFrame(data)
        
        # Initialize categorizer
        categorizer = NLPVideoCategorizer(use_spacy=False)  # Use NLTK for simplicity
        
        # Train model
        results = categorizer.train_model(df, classifier_type='naive_bayes')
        
        print("Training Results:")
        print(f"Accuracy: {results['test_accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Meets Threshold: {results['meets_threshold']}")
        
        # Make predictions
        df_with_predictions = categorizer.predict_categories(df)
        
        # Evaluate predictions
        evaluation = categorizer.evaluate_predictions(df_with_predictions)
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
        print(f"F1 Score: {evaluation['f1_score']:.4f}")
        
        # Save model
        categorizer.save_model()
        
        print("\nNLP categorization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
