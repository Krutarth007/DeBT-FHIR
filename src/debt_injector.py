"""
debt_injector.py - Inject interoperability debt into FHIR data
"""

import numpy as np
import random

class InteroperabilityDebtInjector:
    """Inject various types of interoperability debt into FHIR data"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    def inject_missingness(self, X, intensity):
        """Inject random missing values"""
        X_debt = X.copy()
        n_samples, n_features = X_debt.shape
        
        missing_rate = intensity * 0.7
        for i in range(n_features):
            feature_missing_rate = missing_rate * np.random.uniform(0.7, 1.3)
            missing_mask = np.random.rand(n_samples) < feature_missing_rate
            X_debt[missing_mask, i] = np.nan
        
        return X_debt
    
    def inject_systematic_missing(self, X, intensity):
        """Inject systematic missing values"""
        X_debt = X.copy()
        n_samples, n_features = X_debt.shape
        
        n_affected = max(1, int(n_features * 0.3))
        affected_features = np.random.choice(n_features, n_affected, replace=False)
        missing_rate = intensity * 0.8
        missing_mask = np.random.rand(n_samples) < missing_rate
        
        for f in affected_features:
            X_debt[missing_mask, f] = np.nan
        
        return X_debt
    
    def inject_noise(self, X, intensity):
        """Inject Gaussian noise"""
        X_debt = X.copy()
        n_samples, n_features = X_debt.shape
        
        noise_scale = intensity * 0.5
        for i in range(n_features):
            feature_std = np.std(X_debt[:, i]) if np.std(X_debt[:, i]) > 0 else 1.0
            noise = np.random.normal(0, feature_std * noise_scale, n_samples)
            X_debt[:, i] = X_debt[:, i] + noise
        
        return X_debt
    
    def inject_severe_missingness(self, X, intensity):
        """Inject severe missing values"""
        X_debt = X.copy()
        n_samples, n_features = X_debt.shape
        
        n_affected = max(1, int(n_features * 0.5))
        affected_features = np.random.choice(n_features, n_affected, replace=False)
        
        for f in affected_features:
            missing_mask = np.random.rand(n_samples) < 0.8
            X_debt[missing_mask, f] = np.nan
        
        return X_debt
    
    def inject_outliers(self, X, intensity):
        """Inject outliers"""
        X_debt = X.copy()
        n_samples, n_features = X_debt.shape
        
        for i in range(n_features):
            if np.std(X_debt[:, i]) > 0:
                outlier_rate = intensity * 0.2
                outlier_mask = np.random.rand(n_samples) < outlier_rate
                outlier_shift = np.random.normal(0, 5 * np.std(X_debt[:, i]), np.sum(outlier_mask))
                X_debt[outlier_mask, i] = X_debt[outlier_mask, i] + outlier_shift
        
        return X_debt
    
    def inject_feature_collapse(self, X, intensity):
        """Collapse continuous features into discrete bins"""
        X_debt = X.copy()
        n_features = X_debt.shape[1]
        
        for i in range(n_features):
            if np.std(X_debt[:, i]) > 0 and np.unique(X_debt[:, i]).shape[0] > 10:
                n_bins = max(2, int(10 * (1 - intensity)))
                X_debt[:, i] = np.digitize(X_debt[:, i], 
                                          np.percentile(X_debt[:, i], 
                                                       np.linspace(0, 100, n_bins + 1)[1:-1]))
        
        return X_debt
    
    def impute_with_noise(self, X, intensity):
        """Impute missing values with noisy imputation"""
        X_debt = X.copy()
        n_features = X_debt.shape[1]
        
        for i in range(n_features):
            col_data = X_debt[:, i]
            nan_indices = np.isnan(col_data)
            
            if np.sum(nan_indices) > 0:
                col_mean = np.nanmean(col_data)
                col_std = np.nanstd(col_data) if not np.isnan(np.nanstd(col_data)) else 1.0
                imputation_noise = np.random.normal(0, col_std * intensity * 0.5, np.sum(nan_indices))
                X_debt[nan_indices, i] = col_mean + imputation_noise
        
        return X_debt
    
    def inject_fhir_interoperability_debt(self, fhir_features, intensity, feature_names=None):
        """
        Enhanced debt injection with STRONGER effects for statistical significance
        """
        if intensity == 0:
            return fhir_features.copy()
        
        X = fhir_features.copy()
        
        # Choose debt type based on intensity
        if intensity < 0.3:
            debt_type = 'missingness'
        elif intensity < 0.7:
            debt_type = random.choice(['systematic_missing', 'noise'])
        else:
            debt_type = random.choice(['severe_missingness', 'outlier_injection', 'feature_collapse'])
        
        # Apply selected debt type
        if debt_type == 'missingness':
            X = self.inject_missingness(X, intensity)
        elif debt_type == 'systematic_missing':
            X = self.inject_systematic_missing(X, intensity)
        elif debt_type == 'noise':
            X = self.inject_noise(X, intensity)
        elif debt_type == 'severe_missingness':
            X = self.inject_severe_missingness(X, intensity)
        elif debt_type == 'outlier_injection':
            X = self.inject_outliers(X, intensity)
        elif debt_type == 'feature_collapse':
            X = self.inject_feature_collapse(X, intensity)
        
        # Apply noisy imputation
        X = self.impute_with_noise(X, intensity)
        
        return X
    
    def inject_debt_at_levels(self, X, debt_levels):
        """
        Inject debt at multiple intensity levels
        """
        results = {}
        for intensity in debt_levels:
            X_debt = self.inject_fhir_interoperability_debt(X, intensity)
            results[intensity] = X_debt
        
        return results

if __name__ == "__main__":
    # Test the debt injector
    injector = InteroperabilityDebtInjector()
    
    # Create sample data
    sample_data = np.random.randn(100, 10)
    
    # Test different debt levels
    debt_levels = [0.0, 0.3, 0.7, 1.0]
    
    print("Testing debt injection...")
    for level in debt_levels:
        data_with_debt = injector.inject_fhir_interoperability_debt(sample_data, level)
        print(f"Debt level {level:.1f}: Shape = {data_with_debt.shape}, " 
              f"NaN count = {np.sum(np.isnan(data_with_debt))}")
