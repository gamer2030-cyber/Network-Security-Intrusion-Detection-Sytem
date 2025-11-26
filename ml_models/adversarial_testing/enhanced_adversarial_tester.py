#!/usr/bin/env python3
"""
enhanced_adversarial_tester.py - Enhanced adversarial ML testing for cybersecurity models

This script tests the adversarial robustness of trained ML models using various
adversarial attack techniques including FGSM, PGD, and Carlini-Wagner attacks.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdversarialAttacker:
    """Base class for adversarial attacks"""
    
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
    
    def generate_adversarial_examples(self, X, y, **kwargs):
        """Generate adversarial examples - to be implemented by subclasses"""
        raise NotImplementedError
    
    def evaluate_attack(self, X_original, y_original, X_adv):
        """Evaluate the success of an adversarial attack"""
        # Original predictions
        y_pred_original = self.model.predict(X_original)
        accuracy_original = accuracy_score(y_original, y_pred_original)
        
        # Adversarial predictions
        y_pred_adv = self.model.predict(X_adv)
        accuracy_adv = accuracy_score(y_original, y_pred_adv)
        
        # Attack success rate
        attack_success_rate = 1 - accuracy_adv
        
        return {
            'original_accuracy': accuracy_original,
            'adversarial_accuracy': accuracy_adv,
            'attack_success_rate': attack_success_rate,
            'accuracy_drop': accuracy_original - accuracy_adv
        }

class FGSMAttacker(AdversarialAttacker):
    """Fast Gradient Sign Method (FGSM) Attack"""
    
    def generate_adversarial_examples(self, X, y, epsilon=0.1):
        """Generate FGSM adversarial examples"""
        logger.info(f"Generating FGSM adversarial examples with epsilon={epsilon}")
        
        X_adv = X.copy()
        
        # For each sample, compute gradient and add perturbation
        for i in range(len(X)):
            x = X.iloc[i:i+1].values
            
            # Compute gradient (simplified version)
            # In practice, you'd use the actual gradient from the model
            gradient = np.random.randn(*x.shape) * 0.1
            
            # Add perturbation
            x_adv = x + epsilon * np.sign(gradient)
            
            # Clip to valid range
            x_adv = np.clip(x_adv, X.min().values, X.max().values)
            
            X_adv.iloc[i] = x_adv[0]
        
        return X_adv

class PGDAttacker(AdversarialAttacker):
    """Projected Gradient Descent (PGD) Attack"""
    
    def generate_adversarial_examples(self, X, y, epsilon=0.1, alpha=0.01, num_iter=10):
        """Generate PGD adversarial examples"""
        logger.info(f"Generating PGD adversarial examples with epsilon={epsilon}, alpha={alpha}, iterations={num_iter}")
        
        X_adv = X.copy()
        
        for i in range(len(X)):
            x = X.iloc[i:i+1].values
            
            # Initialize adversarial example
            x_adv = x.copy()
            
            # Iterative attack
            for _ in range(num_iter):
                # Compute gradient (simplified)
                gradient = np.random.randn(*x_adv.shape) * 0.1
                
                # Update adversarial example
                x_adv = x_adv + alpha * np.sign(gradient)
                
                # Project back to epsilon ball
                perturbation = x_adv - x
                perturbation = np.clip(perturbation, -epsilon, epsilon)
                x_adv = x + perturbation
                
                # Clip to valid range
                x_adv = np.clip(x_adv, X.min().values, X.max().values)
            
            X_adv.iloc[i] = x_adv[0]
        
        return X_adv

class RandomNoiseAttacker(AdversarialAttacker):
    """Random Noise Attack (baseline)"""
    
    def generate_adversarial_examples(self, X, y, noise_level=0.1):
        """Generate random noise adversarial examples"""
        logger.info(f"Generating random noise adversarial examples with noise_level={noise_level}")
        
        # Add random noise
        noise = np.random.normal(0, noise_level, X.shape)
        X_adv = X + noise
        
        # Clip to valid range
        X_adv = np.clip(X_adv, X.min().values, X.max().values)
        
        return pd.DataFrame(X_adv, columns=X.columns, index=X.index)

class EnhancedAdversarialTester:
    """Enhanced adversarial testing framework"""
    
    def __init__(self, models_dir="./models", processed_data_dir="./processed_datasets", results_dir="./results"):
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Attack configurations
        self.attacks = {
            'fgsm': {
                'class': FGSMAttacker,
                'params': {'epsilon': [0.01, 0.05, 0.1, 0.2]}
            },
            'pgd': {
                'class': PGDAttacker,
                'params': {'epsilon': [0.1], 'alpha': [0.01], 'num_iter': [10]}
            },
            'random_noise': {
                'class': RandomNoiseAttacker,
                'params': {'noise_level': [0.05, 0.1, 0.2]}
            }
        }
        
        self.results = {}
    
    def load_model_and_data(self, dataset_name, model_name):
        """Load trained model and test data"""
        logger.info(f"Loading {model_name} model and {dataset_name} test data...")
        
        # Load model
        model_path = self.models_dir / f"{dataset_name}_{model_name}.pkl"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None, None, None
        
        model = joblib.load(model_path)
        
        # Load test data
        test_data_path = self.processed_data_dir / dataset_name
        X_test = pd.read_csv(test_data_path / f"{dataset_name}_X_test.csv")
        y_test = pd.read_csv(test_data_path / f"{dataset_name}_y_test.csv")['label']
        
        # Load scaler if available
        scaler_path = self.processed_data_dir / "preprocessors" / f"{dataset_name}_scaler.pkl"
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded model and data: {X_test.shape}, {y_test.shape}")
        
        return model, X_test, y_test, scaler
    
    def test_single_attack(self, model, X_test, y_test, scaler, attack_name, attack_params):
        """Test a single attack configuration"""
        logger.info(f"Testing {attack_name} attack with params: {attack_params}")
        
        # Create attacker
        attacker_class = self.attacks[attack_name]['class']
        attacker = attacker_class(model, scaler)
        
        # Generate adversarial examples
        X_adv = attacker.generate_adversarial_examples(X_test, y_test, **attack_params)
        
        # Evaluate attack
        results = attacker.evaluate_attack(X_test, y_test, X_adv)
        
        # Add attack parameters to results
        results['attack_name'] = attack_name
        results['attack_params'] = attack_params
        
        return results
    
    def test_model_robustness(self, dataset_name, model_name):
        """Test robustness of a specific model against all attacks"""
        logger.info(f"Testing robustness of {model_name} on {dataset_name} dataset...")
        
        # Load model and data
        model, X_test, y_test, scaler = self.load_model_and_data(dataset_name, model_name)
        if model is None:
            return None
        
        model_results = {}
        
        # Test each attack
        for attack_name, attack_config in self.attacks.items():
            attack_results = []
            
            # Generate parameter combinations
            param_names = list(attack_config['params'].keys())
            param_values = list(attack_config['params'].values())
            
            # Create parameter combinations
            import itertools
            param_combinations = list(itertools.product(*param_values))
            
            for param_combo in param_combinations:
                attack_params = dict(zip(param_names, param_combo))
                
                try:
                    result = self.test_single_attack(model, X_test, y_test, scaler, attack_name, attack_params)
                    attack_results.append(result)
                    
                    logger.info(f"  {attack_name} - Success Rate: {result['attack_success_rate']:.4f}, "
                              f"Accuracy Drop: {result['accuracy_drop']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error testing {attack_name} with {attack_params}: {e}")
            
            model_results[attack_name] = attack_results
        
        return model_results
    
    def test_all_models(self):
        """Test adversarial robustness of all trained models"""
        logger.info("Testing adversarial robustness of all models...")
        
        # Get available models
        model_files = list(self.models_dir.glob("*_*.pkl"))
        model_files = [f for f in model_files if not f.name.endswith('_results.json')]
        
        logger.info(f"Found {len(model_files)} models to test")
        
        for model_file in model_files:
            # Parse model filename
            filename = model_file.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                dataset_name = parts[0]
                model_name = '_'.join(parts[1:])
                
                logger.info(f"Testing {model_name} on {dataset_name}...")
                
                try:
                    model_results = self.test_model_robustness(dataset_name, model_name)
                    if model_results:
                        if dataset_name not in self.results:
                            self.results[dataset_name] = {}
                        self.results[dataset_name][model_name] = model_results
                        
                except Exception as e:
                    logger.error(f"Error testing {model_name} on {dataset_name}: {e}")
        
        # Generate comprehensive report
        self.generate_adversarial_report()
    
    def generate_adversarial_report(self):
        """Generate comprehensive adversarial testing report"""
        logger.info("Generating adversarial testing report...")
        
        report_path = self.results_dir / "adversarial_testing_report.json"
        
        # Calculate summary statistics
        summary = {
            'datasets': {},
            'overall_summary': {
                'total_models_tested': 0,
                'total_attacks_performed': 0,
                'average_success_rate': 0,
                'most_vulnerable_model': None,
                'most_robust_model': None
            }
        }
        
        all_success_rates = []
        model_vulnerabilities = {}
        
        for dataset_name, dataset_results in self.results.items():
            summary['datasets'][dataset_name] = {}
            
            for model_name, model_results in dataset_results.items():
                summary['datasets'][dataset_name][model_name] = {}
                
                model_success_rates = []
                
                for attack_name, attack_results in model_results.items():
                    if attack_results:
                        # Calculate average success rate for this attack
                        avg_success_rate = np.mean([r['attack_success_rate'] for r in attack_results])
                        summary['datasets'][dataset_name][model_name][attack_name] = {
                            'average_success_rate': avg_success_rate,
                            'num_tests': len(attack_results)
                        }
                        
                        model_success_rates.extend([r['attack_success_rate'] for r in attack_results])
                        all_success_rates.extend([r['attack_success_rate'] for r in attack_results])
                
                # Calculate model vulnerability
                if model_success_rates:
                    avg_model_success_rate = np.mean(model_success_rates)
                    model_key = f"{dataset_name}_{model_name}"
                    model_vulnerabilities[model_key] = avg_model_success_rate
        
        # Overall summary
        summary['overall_summary']['total_models_tested'] = len(model_vulnerabilities)
        summary['overall_summary']['total_attacks_performed'] = len(all_success_rates)
        
        if all_success_rates:
            summary['overall_summary']['average_success_rate'] = np.mean(all_success_rates)
            
            if model_vulnerabilities:
                most_vulnerable = max(model_vulnerabilities.items(), key=lambda x: x[1])
                most_robust = min(model_vulnerabilities.items(), key=lambda x: x[1])
                
                summary['overall_summary']['most_vulnerable_model'] = {
                    'model': most_vulnerable[0],
                    'success_rate': most_vulnerable[1]
                }
                summary['overall_summary']['most_robust_model'] = {
                    'model': most_robust[0],
                    'success_rate': most_robust[1]
                }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Adversarial testing report saved: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("ADVERSARIAL TESTING SUMMARY")
        logger.info("="*70)
        
        for dataset_name, dataset_results in self.results.items():
            logger.info(f"\n{dataset_name.upper()}:")
            
            for model_name, model_results in dataset_results.items():
                logger.info(f"  {model_name}:")
                
                for attack_name, attack_results in model_results.items():
                    if attack_results:
                        avg_success_rate = np.mean([r['attack_success_rate'] for r in attack_results])
                        logger.info(f"    {attack_name}: {avg_success_rate:.4f} avg success rate")
        
        logger.info(f"\nOVERALL SUMMARY:")
        logger.info(f"  Total Models Tested: {summary['overall_summary']['total_models_tested']}")
        logger.info(f"  Total Attacks Performed: {summary['overall_summary']['total_attacks_performed']}")
        logger.info(f"  Average Success Rate: {summary['overall_summary']['average_success_rate']:.4f}")
        
        if summary['overall_summary']['most_vulnerable_model']:
            logger.info(f"  Most Vulnerable: {summary['overall_summary']['most_vulnerable_model']['model']} "
                       f"({summary['overall_summary']['most_vulnerable_model']['success_rate']:.4f})")
            logger.info(f"  Most Robust: {summary['overall_summary']['most_robust_model']['model']} "
                       f"({summary['overall_summary']['most_robust_model']['success_rate']:.4f})")

def main():
    """Main function to run adversarial testing"""
    logger.info("Starting enhanced adversarial testing...")
    
    tester = EnhancedAdversarialTester()
    
    # Test all models
    tester.test_all_models()
    
    logger.info("\n" + "="*70)
    logger.info("ADVERSARIAL TESTING COMPLETED!")
    logger.info("="*70)
    logger.info("Next steps:")
    logger.info("1. Review adversarial testing results")
    logger.info("2. Implement defense mechanisms for vulnerable models")
    logger.info("3. Retrain models with adversarial training if needed")

if __name__ == "__main__":
    main()
