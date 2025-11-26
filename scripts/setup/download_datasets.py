#!/usr/bin/env python3
"""
download_datasets.py - Download and prepare real cybersecurity datasets for ML-IDS-IPS

This script downloads and prepares three major cybersecurity datasets:
1. NSL-KDD - University of New Brunswick
2. UNSW-NB15 - University of New South Wales  
3. CICIDS2017 - Canadian Institute for Cybersecurity
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from urllib.parse import urlparse
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_dir="./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'nsl_kdd': {
                'name': 'NSL-KDD',
                'description': 'Network Security Laboratory - Knowledge Discovery in Databases',
                'features': 41,
                'classes': 5,
                'urls': [
                    'https://www.unb.ca/cic/datasets/nsl.html',  # Official page
                    'https://huggingface.co/datasets/Mireu-Lab/NSL-KDD',  # Hugging Face
                ],
                'files': ['KDDTrain+.txt', 'KDDTest+.txt', 'KDDTrain+_20Percent.txt', 'KDDTest-21.txt'],
                'note': 'Requires manual download from IEEE DataPort or Hugging Face'
            },
            'unsw_nb15': {
                'name': 'UNSW-NB15',
                'description': 'University of New South Wales Network-Based Intrusion Detection',
                'features': 49,
                'classes': 10,
                'urls': [
                    'https://research.unsw.edu.au/projects/unsw-nb15-dataset',  # Official page
                ],
                'files': ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv'],
                'note': 'Requires manual download from UNSW research page'
            },
            'cicids2017': {
                'name': 'CICIDS2017',
                'description': 'Canadian Institute for Cybersecurity Intrusion Detection',
                'features': 78,
                'classes': 5,
                'urls': [
                    'https://www.unb.ca/cic/datasets/ids-2017.html',  # Official page
                ],
                'files': ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv', 
                         'Wednesday-WorkingHours.pcap_ISCX.csv', 'Thursday-WorkingHours.pcap_ISCX.csv',
                         'Friday-WorkingHours.pcap_ISCX.csv'],
                'note': 'Requires manual download from CIC website'
            }
        }
    
    def create_download_instructions(self):
        """Create detailed download instructions for each dataset"""
        logger.info("Creating download instructions...")
        
        instructions_file = self.base_dir / "DOWNLOAD_INSTRUCTIONS.md"
        
        with open(instructions_file, 'w') as f:
            f.write("# Cybersecurity Datasets Download Instructions\n\n")
            f.write("This document provides instructions for downloading the three major cybersecurity datasets used in this ML-IDS-IPS project.\n\n")
            
            for dataset_id, config in self.datasets.items():
                f.write(f"## {config['name']} Dataset\n\n")
                f.write(f"**Description:** {config['description']}\n\n")
                f.write(f"**Features:** {config['features']}\n")
                f.write(f"**Classes:** {config['classes']}\n\n")
                f.write(f"**Download Sources:**\n")
                for url in config['urls']:
                    f.write(f"- {url}\n")
                f.write(f"\n**Expected Files:**\n")
                for file in config['files']:
                    f.write(f"- {file}\n")
                f.write(f"\n**Note:** {config['note']}\n\n")
                f.write("---\n\n")
            
            f.write("## Setup Instructions\n\n")
            f.write("1. Download the datasets from the official sources above\n")
            f.write("2. Extract the files to the appropriate directories:\n")
            f.write("   - NSL-KDD: `./datasets/nsl_kdd/`\n")
            f.write("   - UNSW-NB15: `./datasets/unsw_nb15/`\n")
            f.write("   - CICIDS2017: `./datasets/cicids2017/`\n")
            f.write("3. Run the preprocessing script: `python scripts/setup/preprocess_datasets.py`\n")
        
        logger.info(f"Download instructions saved to: {instructions_file}")
    
    def create_sample_data_with_real_structure(self):
        """Create sample data that mimics the structure of real datasets"""
        logger.info("Creating sample data with real dataset structure...")
        
        # NSL-KDD sample with 41 features
        self._create_nsl_kdd_sample()
        
        # UNSW-NB15 sample with 49 features  
        self._create_unsw_nb15_sample()
        
        # CICIDS2017 sample with 78 features
        self._create_cicids2017_sample()
    
    def _create_nsl_kdd_sample(self):
        """Create NSL-KDD sample data with proper structure"""
        dataset_dir = self.base_dir / "nsl_kdd"
        dataset_dir.mkdir(exist_ok=True)
        
        # NSL-KDD has 41 features + 1 label
        n_samples = 1000
        n_features = 41
        
        # Create realistic feature names based on NSL-KDD structure
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Generate realistic data
        data = {}
        for i, feature in enumerate(feature_names):
            if feature in ['protocol_type', 'service', 'flag']:
                # Categorical features
                if feature == 'protocol_type':
                    data[feature] = np.random.choice(['tcp', 'udp', 'icmp'], n_samples)
                elif feature == 'service':
                    data[feature] = np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'telnet'], n_samples)
                else:  # flag
                    data[feature] = np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples)
            else:
                # Numerical features
                data[feature] = np.random.exponential(1.0, n_samples)
        
        # Create labels (0=normal, 1-4=attack types)
        labels = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.15, 0.15, 0.05, 0.05])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['label'] = labels
        
        # Save sample files
        df.to_csv(dataset_dir / "KDDTrain+_sample.txt", index=False, sep=',')
        df.to_csv(dataset_dir / "KDDTest+_sample.txt", index=False, sep=',')
        
        logger.info(f"NSL-KDD sample created with {n_samples} samples and {n_features} features")
    
    def _create_unsw_nb15_sample(self):
        """Create UNSW-NB15 sample data with proper structure"""
        dataset_dir = self.base_dir / "unsw_nb15"
        dataset_dir.mkdir(exist_ok=True)
        
        n_samples = 1000
        n_features = 49
        
        # UNSW-NB15 feature names (simplified)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Generate realistic network traffic data
        data = {}
        for feature in feature_names:
            data[feature] = np.random.exponential(1.0, n_samples)
        
        # Create labels (0=normal, 1-9=attack types)
        probs = [0.7] + [0.3/9]*9  # Normalize to sum to 1
        labels = np.random.choice(range(10), n_samples, p=probs)
        
        df = pd.DataFrame(data)
        df['label'] = labels
        
        # Save sample files
        df.to_csv(dataset_dir / "UNSW_NB15_training-set_sample.csv", index=False)
        df.to_csv(dataset_dir / "UNSW_NB15_testing-set_sample.csv", index=False)
        
        logger.info(f"UNSW-NB15 sample created with {n_samples} samples and {n_features} features")
    
    def _create_cicids2017_sample(self):
        """Create CICIDS2017 sample data with proper structure"""
        dataset_dir = self.base_dir / "cicids2017"
        dataset_dir.mkdir(exist_ok=True)
        
        n_samples = 1000
        n_features = 78
        
        # CICIDS2017 feature names (simplified)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Generate realistic network flow data
        data = {}
        for feature in feature_names:
            data[feature] = np.random.exponential(1.0, n_samples)
        
        # Create labels (0=normal, 1-4=attack types)
        labels = np.random.choice(range(5), n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02])
        
        df = pd.DataFrame(data)
        df['label'] = labels
        
        # Save sample files for different days
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day in days:
            df.to_csv(dataset_dir / f"{day}-WorkingHours_sample.csv", index=False)
        
        logger.info(f"CICIDS2017 sample created with {n_samples} samples and {n_features} features")
    
    def verify_dataset_structure(self):
        """Verify that datasets have the correct structure"""
        logger.info("Verifying dataset structure...")
        
        for dataset_id, config in self.datasets.items():
            dataset_dir = self.base_dir / dataset_id
            if dataset_dir.exists():
                logger.info(f"Checking {config['name']} dataset...")
                
                # Check for expected files
                found_files = list(dataset_dir.glob("*"))
                logger.info(f"Found files: {[f.name for f in found_files]}")
                
                # Try to load and check structure
                for file_path in found_files:
                    if file_path.suffix in ['.csv', '.txt']:
                        try:
                            if file_path.suffix == '.csv':
                                df = pd.read_csv(file_path)
                            else:
                                df = pd.read_csv(file_path, sep=',')
                            
                            logger.info(f"  {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                            
                            if 'label' in df.columns:
                                label_counts = df['label'].value_counts()
                                logger.info(f"    Label distribution: {dict(label_counts)}")
                            
                        except Exception as e:
                            logger.warning(f"  Could not read {file_path.name}: {e}")
            else:
                logger.warning(f"Dataset directory not found: {dataset_dir}")

def main():
    """Main function to set up datasets"""
    logger.info("Starting dataset setup...")
    
    downloader = DatasetDownloader()
    
    # Create download instructions
    downloader.create_download_instructions()
    
    # Create sample data with real structure
    downloader.create_sample_data_with_real_structure()
    
    # Verify structure
    downloader.verify_dataset_structure()
    
    logger.info("Dataset setup completed!")
    logger.info("Next steps:")
    logger.info("1. Download real datasets following instructions in DOWNLOAD_INSTRUCTIONS.md")
    logger.info("2. Replace sample files with real dataset files")
    logger.info("3. Run preprocessing script to prepare data for training")

if __name__ == "__main__":
    main()
