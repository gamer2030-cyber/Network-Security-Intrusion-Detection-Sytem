#!/usr/bin/env python3
# adversarial_tester.py - Adversarial ML testing

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialTester:
    def __init__(self, results_dir="./results"):
        self.results_dir = results_dir
        self.results = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def create_sample_model(self, n_samples=1000, n_features=20):
        """Create a sample trained model"""
        logger.info("Creating sample trained model...")
        
        # Generate data
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_fgsm_simulation(self, model, X_test, y_test, epsilon=0.1):
        """Simulate FGSM attack"""
        logger.info(f"Testing FGSM attack with epsilon={epsilon}")
        
        # Simulate adversarial examples
        X_adv = X_test + epsilon * np.random.randn(*X_test.shape)
        
        # Evaluate model
        predictions = model.predict(X_adv)
        accuracy = np.mean(predictions == y_test)
        
        result = {
            'attack_type': 'fgsm',
            'epsilon': epsilon,
            'accuracy': accuracy,
            'success_rate': 1 - accuracy
        }
        
        self.results['fgsm'] = result
        logger.info(f"FGSM attack success rate: {result['success_rate']:.4f}")
        
        return result
    
    def test_pgd_simulation(self, model, X_test, y_test, epsilon=0.1, iterations=10):
        """Simulate PGD attack"""
        logger.info(f"Testing PGD attack with epsilon={epsilon}, iterations={iterations}")
        
        X_adv = X_test.copy()
        
        # Simulate iterative attack
        for i in range(iterations):
            X_adv += (epsilon / iterations) * np.random.randn(*X_adv.shape)
        
        # Evaluate model
        predictions = model.predict(X_adv)
        accuracy = np.mean(predictions == y_test)
        
        result = {
            'attack_type': 'pgd',
            'epsilon': epsilon,
            'iterations': iterations,
            'accuracy': accuracy,
            'success_rate': 1 - accuracy
        }
        
        self.results['pgd'] = result
        logger.info(f"PGD attack success rate: {result['success_rate']:.4f}")
        
        return result
    
    def generate_report(self):
        """Generate adversarial testing report"""
        logger.info("Generating adversarial testing report...")
        
        report = {
            'summary': {
                'total_attacks': len(self.results),
                'results': self.results
            }
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, 'adversarial_test_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report

if __name__ == "__main__":
    tester = AdversarialTester()
    
    # Create sample model
    model, X_test, y_test = tester.create_sample_model()
    
    # Test attacks
    fgsm_result = tester.test_fgsm_simulation(model, X_test, y_test)
    pgd_result = tester.test_pgd_simulation(model, X_test, y_test)
    
    # Generate report
    report = tester.generate_report()
    
    print(f"\nAdversarial testing completed!")
    print(f"FGSM attack success rate: {fgsm_result['success_rate']:.4f}")
    print(f"PGD attack success rate: {pgd_result['success_rate']:.4f}")
