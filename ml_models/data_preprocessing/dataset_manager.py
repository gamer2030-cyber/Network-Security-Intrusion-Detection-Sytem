#!/usr/bin/env python3
# dataset_manager.py - Dataset management for ML-IDS-IPS

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, base_dir="./datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_sample_dataset(self, dataset_name, n_samples=1000):
        """Create a sample dataset for testing"""
        logger.info(f"Creating sample dataset: {dataset_name}")
        
        # Create sample data
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['label'] = y
        
        # Save dataset
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        df.to_csv(os.path.join(dataset_dir, f'{dataset_name}_sample.csv'), index=False)
        
        logger.info(f"Sample dataset created: {dataset_name}")
        return df

if __name__ == "__main__":
    manager = DatasetManager()
    manager.create_sample_dataset("nsl_kdd")
    manager.create_sample_dataset("unsw_nb15")
    manager.create_sample_dataset("cicids2017")
    print("Sample datasets created successfully!")
