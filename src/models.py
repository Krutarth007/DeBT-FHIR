"""
models.py - Machine learning models for evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix, 
                           brier_score_loss, average_precision_score)

class EnsembleModel:
    """Ensemble of Logistic Regression and Random Forest"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.models = [
            LogisticRegression(class_weight='balanced', max_iter=1000, 
                             random_state=random_seed),
            RandomForestClassifier(n_estimators=50, max_depth=5,
                                 class_weight='balanced',
                                 random_state=random_seed)
        ]
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit all models in the ensemble"""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
    
    def predict_proba(self, X):
        """Get probability predictions from ensemble average"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        ensemble_predictions = []
        for model in self.models:
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]
                ensemble_predictions.append(y_pred_proba)
            except Exception as e:
                print(f"Warning: Model prediction failed: {e}")
                continue
        
        if not ensemble_predictions:
            raise ValueError("No models produced predictions")
        
        return np.mean(ensemble_predictions, axis=0)
    
    def predict(self, X, threshold=0.5):
        """Get binary predictions"""
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)
    
    def evaluate(self, X, y, threshold=0.5):
        """Comprehensive evaluation"""
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X, threshold)
        
        # Calculate metrics
        auc = roc_auc_score(y, y_pred_proba)
        pr_auc = average_precision_score(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'auc': auc,
            'pr_auc': pr_auc,
            'brier': brier,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

def evaluate_ensemble_cross_validation(X, y, n_splits=5, random_seed=42):
    """
    Evaluate ensemble model using cross-validation
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_results = []
    
    if np.sum(y) < 20:
        return None
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        # Initialize and train ensemble
        ensemble = EnsembleModel(random_seed=random_seed + fold)
        ensemble.fit(X_train, y_train)
        
        # Evaluate on test set
        metrics = ensemble.evaluate(X_test, y_test)
        
        fold_results.append({
            'fold': fold,
            'auc': metrics['auc'],
            'pr_auc': metrics['pr_auc'],
            'brier': metrics['brier'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity']
        })
    
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        return results_df
    else:
        return None

def calculate_bootstrap_ci(X, y, n_bootstraps=300, random_seed=42):
    """
    Calculate bootstrap confidence intervals
    """
    np.random.seed(random_seed)
    auc_scores = []
    n_samples = len(y)
    
    for _ in range(n_bootstraps):
        # Stratified sampling
        indices_0 = np.where(y == 0)[0]
        indices_1 = np.where(y == 1)[0]
        
        bootstrap_indices_0 = np.random.choice(indices_0, size=len(indices_0), replace=True)
        bootstrap_indices_1 = np.random.choice(indices_1, size=len(indices_1), replace=True)
        
        bootstrap_indices = np.concatenate([bootstrap_indices_0, bootstrap_indices_1])
        np.random.shuffle(bootstrap_indices)
        
        X_boot = X[bootstrap_indices]
        y_boot = y[bootstrap_indices]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        # Train and evaluate
        ensemble = EnsembleModel(random_seed=random_seed)
        ensemble.fit(X_boot, y_boot)
        y_pred_proba = ensemble.predict_proba(X_boot)
        auc = roc_auc_score(y_boot, y_pred_proba)
        auc_scores.append(auc)
    
    if auc_scores:
        ci_lower = float(np.percentile(auc_scores, 2.5))
        ci_upper = float(np.percentile(auc_scores, 97.5))
        return {
            'auc_ci_lower': ci_lower,
            'auc_ci_upper': ci_upper,
            'auc_std': float(np.std(auc_scores))
        }
    else:
        return {
            'auc_ci_lower': 0.5,
            'auc_ci_upper': 0.5,
            'auc_std': 0.0
        }

if __name__ == "__main__":
    # Test the models
    print("Testing ensemble model...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + np.random.randn(100) > 0).astype(int)
    
    # Test ensemble
    ensemble = EnsembleModel()
    ensemble.fit(X, y)
    
    metrics = ensemble.evaluate(X, y)
    print(f"AUC: {metrics['auc']:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
