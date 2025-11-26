#!/usr/bin/env python3
# model_trainer.py - ML model training for IDS/IPS

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def create_sample_data(self, n_samples=1000, n_features=20):
        """Create sample training data"""
        logger.info("Creating sample training data...")
        
        # Generate synthetic data
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "random_forest.pkl")
        joblib.dump(model, model_path)
        
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model"""
        logger.info("Training SVM model...")
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"SVM accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "svm.pkl")
        joblib.dump(model, model_path)
        
        return model, accuracy

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Create sample data
    X_train, X_test, y_train, y_test = trainer.create_sample_data()
    
    # Train models
    rf_model, rf_acc = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    svm_model, svm_acc = trainer.train_svm(X_train, y_train, X_test, y_test)
    
    print(f"\nTraining completed!")
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print(f"SVM accuracy: {svm_acc:.4f}")
