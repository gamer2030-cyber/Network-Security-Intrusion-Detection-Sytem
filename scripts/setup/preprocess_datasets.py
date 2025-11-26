#!/usr/bin/env python3
"""
preprocess_datasets.py - Enhanced data preprocessing for cybersecurity datasets

This script preprocesses the cybersecurity datasets (NSL-KDD, UNSW-NB15, CICIDS2017)
with proper feature engineering, encoding, and normalization for ML training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetPreprocessor:
    def __init__(self, datasets_dir="./datasets", output_dir="./processed_datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.feature_scalers = {}
        self.feature_encoders = {}
        
        # Dataset configurations
        self.dataset_configs = {
            'nsl_kdd': {
                'name': 'NSL-KDD',
                'categorical_features': ['protocol_type', 'service', 'flag'],
                'target_column': 'label',
                'train_file': 'KDDTrain+_sample.txt',
                'test_file': 'KDDTest+_sample.txt'
            },
            'unsw_nb15': {
                'name': 'UNSW-NB15',
                'categorical_features': [],  # Will be determined dynamically
                'target_column': 'label',
                'train_file': 'UNSW_NB15_training-set_sample.csv',
                'test_file': 'UNSW_NB15_testing-set_sample.csv'
            },
            'cicids2017': {
                'name': 'CICIDS2017',
                'categorical_features': [],  # Will be determined dynamically
                'target_column': 'label',
                'train_files': ['Monday-WorkingHours_sample.csv', 'Tuesday-WorkingHours_sample.csv',
                               'Wednesday-WorkingHours_sample.csv', 'Thursday-WorkingHours_sample.csv',
                               'Friday-WorkingHours_sample.csv']
            }
        }
    
    def load_dataset(self, dataset_name):
        """Load dataset based on its type"""
        config = self.dataset_configs[dataset_name]
        dataset_path = self.datasets_dir / dataset_name
        
        logger.info(f"Loading {config['name']} dataset...")
        
        if dataset_name == 'cicids2017':
            # Load multiple files for CICIDS2017
            dataframes = []
            for file_name in config['train_files']:
                file_path = dataset_path / file_name
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                    logger.info(f"Loaded {file_name}: {df.shape}")
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                logger.info(f"Combined CICIDS2017 data: {combined_df.shape}")
                return combined_df
            else:
                logger.warning("No CICIDS2017 files found")
                return None
        
        else:
            # Load single file datasets
            train_file = dataset_path / config['train_file']
            if train_file.exists():
                df = pd.read_csv(train_file)
                logger.info(f"Loaded {config['train_file']}: {df.shape}")
                return df
            else:
                logger.warning(f"File not found: {train_file}")
                return None
    
    def detect_categorical_features(self, df, dataset_name):
        """Automatically detect categorical features"""
        categorical_features = []
        
        for column in df.columns:
            if column == self.dataset_configs[dataset_name]['target_column']:
                continue
            
            # Check if column is categorical
            if df[column].dtype == 'object' or df[column].nunique() < 20:
                categorical_features.append(column)
        
        logger.info(f"Detected categorical features: {categorical_features}")
        return categorical_features
    
    def preprocess_dataset(self, dataset_name):
        """Preprocess a specific dataset"""
        logger.info(f"Preprocessing {dataset_name} dataset...")
        
        # Load dataset
        df = self.load_dataset(dataset_name)
        if df is None:
            logger.error(f"Failed to load {dataset_name} dataset")
            return None
        
        # Detect categorical features if not specified
        config = self.dataset_configs[dataset_name]
        if not config['categorical_features']:
            config['categorical_features'] = self.detect_categorical_features(df, dataset_name)
        
        # Separate features and target
        target_column = config['target_column']
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return None
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Handle categorical features
        categorical_features = config['categorical_features']
        for feature in categorical_features:
            if feature in X.columns:
                # Encode categorical features
                if feature not in self.feature_encoders:
                    self.feature_encoders[feature] = LabelEncoder()
                    X[feature] = self.feature_encoders[feature].fit_transform(X[feature].astype(str))
                else:
                    X[feature] = self.feature_encoders[feature].transform(X[feature].astype(str))
        
        # Encode target labels
        if dataset_name not in self.label_encoders:
            self.label_encoders[dataset_name] = LabelEncoder()
            y_encoded = self.label_encoders[dataset_name].fit_transform(y)
        else:
            y_encoded = self.label_encoders[dataset_name].transform(y)
        
        # Scale features
        scaler_name = f"{dataset_name}_scaler"
        if scaler_name not in self.feature_scalers:
            self.feature_scalers[scaler_name] = StandardScaler()
            X_scaled = self.feature_scalers[scaler_name].fit_transform(X)
        else:
            X_scaled = self.feature_scalers[scaler_name].transform(X)
        
        # Convert back to DataFrame
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Save processed data
        output_path = self.output_dir / dataset_name
        output_path.mkdir(exist_ok=True)
        
        # Save training data
        X_train.to_csv(output_path / f"{dataset_name}_X_train.csv", index=False)
        y_train_df = pd.DataFrame(y_train, columns=['label'])
        y_train_df.to_csv(output_path / f"{dataset_name}_y_train.csv", index=False)
        
        # Save test data
        X_test.to_csv(output_path / f"{dataset_name}_X_test.csv", index=False)
        y_test_df = pd.DataFrame(y_test, columns=['label'])
        y_test_df.to_csv(output_path / f"{dataset_name}_y_test.csv", index=False)
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'original_shape': list(df.shape),
            'processed_shape': list(X_scaled_df.shape),
            'categorical_features': categorical_features,
            'train_shape': list(X_train.shape),
            'test_shape': list(X_test.shape),
            'label_mapping': dict(zip(
                self.label_encoders[dataset_name].classes_.astype(str),
                range(len(self.label_encoders[dataset_name].classes_))
            )),
            'feature_names': list(X.columns)
        }
        
        with open(output_path / f"{dataset_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Processed {dataset_name} dataset saved to {output_path}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'metadata': metadata
        }
    
    def save_preprocessors(self):
        """Save all preprocessors for later use"""
        preprocessors_dir = self.output_dir / "preprocessors"
        preprocessors_dir.mkdir(exist_ok=True)
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, preprocessors_dir / f"{name}_label_encoder.pkl")
        
        # Save feature scalers
        for name, scaler in self.feature_scalers.items():
            joblib.dump(scaler, preprocessors_dir / f"{name}_scaler.pkl")
        
        # Save feature encoders
        for name, encoder in self.feature_encoders.items():
            joblib.dump(encoder, preprocessors_dir / f"{name}_feature_encoder.pkl")
        
        logger.info(f"Preprocessors saved to {preprocessors_dir}")
    
    def process_all_datasets(self):
        """Process all available datasets"""
        logger.info("Processing all datasets...")
        
        processed_datasets = {}
        
        for dataset_name in self.dataset_configs.keys():
            try:
                result = self.preprocess_dataset(dataset_name)
                if result:
                    processed_datasets[dataset_name] = result
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
        
        # Save preprocessors
        self.save_preprocessors()
        
        logger.info(f"Successfully processed {len(processed_datasets)} datasets")
        return processed_datasets

def main():
    """Main function to preprocess all datasets"""
    logger.info("Starting dataset preprocessing...")
    
    preprocessor = DatasetPreprocessor()
    
    # Process all datasets
    processed_datasets = preprocessor.process_all_datasets()
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*50)
    
    for dataset_name, data in processed_datasets.items():
        metadata = data['metadata']
        logger.info(f"\n{metadata['dataset_name'].upper()}:")
        logger.info(f"  Original shape: {metadata['original_shape']}")
        logger.info(f"  Processed shape: {metadata['processed_shape']}")
        logger.info(f"  Train samples: {metadata['train_shape'][0]}")
        logger.info(f"  Test samples: {metadata['test_shape'][0]}")
        logger.info(f"  Features: {metadata['processed_shape'][1]}")
        logger.info(f"  Classes: {len(metadata['label_mapping'])}")
        logger.info(f"  Categorical features: {len(metadata['categorical_features'])}")
    
    logger.info("\n" + "="*50)
    logger.info("Next steps:")
    logger.info("1. Train models with processed data")
    logger.info("2. Evaluate model performance")
    logger.info("3. Test adversarial robustness")

if __name__ == "__main__":
    main()
