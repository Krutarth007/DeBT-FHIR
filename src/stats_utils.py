"""
stats_utils.py - Statistical analysis utilities
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

def calculate_spearman_correlation(debt_levels, auc_scores):
    """Calculate Spearman rank correlation"""
    rho, p_value = stats.spearmanr(debt_levels, auc_scores)
    return {'rho': rho, 'p_value': p_value}

def calculate_linear_regression(debt_levels, auc_scores):
    """Calculate linear regression relationship"""
    X = sm.add_constant(debt_levels)  # Add constant for intercept
    model = sm.OLS(auc_scores, X).fit()
    
    return {
        'slope': model.params[1],
        'intercept': model.params[0],
        'p_value': model.pvalues[1],
        'r_squared': model.rsquared,
        'model_summary': str(model.summary())
    }

def calculate_effect_size(baseline_auc, max_debt_auc, pooled_std):
    """Calculate Cohen's d effect size"""
    if pooled_std > 0:
        effect_size = (baseline_auc - max_debt_auc) / pooled_std
    else:
        effect_size = 0
    
    # Interpret effect size
    if abs(effect_size) < 0.2:
        magnitude = 'negligible'
    elif abs(effect_size) < 0.5:
        magnitude = 'small'
    elif abs(effect_size) < 0.8:
        magnitude = 'medium'
    else:
        magnitude = 'large'
    
    return {
        'effect_size': effect_size,
        'magnitude': magnitude
    }

def count_meaningful_degradations(results_df, baseline_auc, auc_ci_width=0.05):
    """
    Count meaningful degradations based on confidence intervals
    """
    meaningful_count = 0
    degradation_details = []
    
    for _, row in results_df.iterrows():
        if row['Debt_Intensity'] == 0:
            continue
        
        degradation = baseline_auc - row['AUC']
        is_meaningful = degradation > auc_ci_width
        
        if is_meaningful:
            meaningful_count += 1
            degradation_details.append({
                'debt_intensity': row['Debt_Intensity'],
                'auc_degradation': degradation,
                'is_meaningful': True
            })
        else:
            degradation_details.append({
                'debt_intensity': row['Debt_Intensity'],
                'auc_degradation': degradation,
                'is_meaningful': False
            })
    
    return {
        'meaningful_count': meaningful_count,
        'total_tested': len(results_df) - 1,  # Exclude baseline
        'degradation_details': degradation_details
    }

def calculate_degradation_threshold(results_df, min_degradation=0.02):
    """
    Find the debt intensity where degradation becomes clinically meaningful
    """
    baseline_row = results_df[results_df['Debt_Intensity'] == 0]
    if baseline_row.empty:
        return None
    
    baseline_auc = baseline_row.iloc[0]['AUC']
    
    # Find first debt level where degradation exceeds threshold
    for _, row in results_df.iterrows():
        if row['Debt_Intensity'] == 0:
            continue
        
        degradation = baseline_auc - row['AUC']
        if degradation >= min_degradation:
            return {
                'threshold_intensity': row['Debt_Intensity'],
                'degradation_at_threshold': degradation,
                'baseline_auc': baseline_auc
            }
    
    return {
        'threshold_intensity': None,
        'degradation_at_threshold': 0,
        'baseline_auc': baseline_auc
    }

def perform_complete_statistical_analysis(results_df):
    """
    Perform complete statistical analysis on results
    """
    analysis_results = {}
    
    # Group by task
    tasks = results_df['Task'].unique()
    
    for task in tasks:
        task_data = results_df[results_df['Task'] == task].sort_values('Debt_Intensity')
        
        if len(task_data) <= 2:
            continue
