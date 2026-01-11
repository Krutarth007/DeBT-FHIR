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
        
        # Extract data
        debt_levels = task_data['Debt_Intensity'].values
        auc_scores = task_data['AUC'].values
        
        # Calculate statistics
        spearman_results = calculate_spearman_correlation(debt_levels, auc_scores)
        regression_results = calculate_linear_regression(debt_levels, auc_scores)
        
        # Calculate effect size
        baseline_auc = task_data[task_data['Debt_Intensity'] == 0]['AUC'].values[0]
        max_debt_auc = task_data[task_data['Debt_Intensity'] == 1.0]
        max_debt_auc = max_debt_auc['AUC'].values[0] if not max_debt_auc.empty else baseline_auc
        
        pooled_std = np.std(auc_scores)
        effect_size_results = calculate_effect_size(baseline_auc, max_debt_auc, pooled_std)
        
        # Count meaningful degradations
        degradation_results = count_meaningful_degradations(task_data, baseline_auc)
        
        # Find degradation threshold
        threshold_results = calculate_degradation_threshold(task_data)
        
        # Combine all results
        analysis_results[task] = {
            'spearman': spearman_results,
            'linear_regression': regression_results,
            'effect_size': effect_size_results,
            'degradation_analysis': degradation_results,
            'threshold_analysis': threshold_results,
            'summary_statistics': {
                'baseline_auc': float(baseline_auc),
                'max_degradation': float(baseline_auc - max_debt_auc),
                'relative_degradation': float((baseline_auc - max_debt_auc) / baseline_auc * 100),
                'n_observations': len(task_data)
            }
        }
    
    return analysis_results

def create_statistical_summary(analysis_results):
    """
    Create a human-readable summary of statistical analysis
    """
    summary = []
    summary.append("=" * 80)
    summary.append("STATISTICAL ANALYSIS SUMMARY")
    summary.append("=" * 80)
    
    for task, results in analysis_results.items():
        summary.append(f"\n{task}:")
        summary.append("-" * 40)
        
        # Spearman correlation
        spearman = results['spearman']
        summary.append(f"Spearman ρ: {spearman['rho']:.3f} (p = {spearman['p_value']:.3e})")
        
        # Linear regression
        regression = results['linear_regression']
        summary.append(f"Linear slope: {regression['slope']:.3f} (p = {regression['p_value']:.3e})")
        summary.append(f"R²: {regression['r_squared']:.3f}")
        
        # Effect size
        effect = results['effect_size']
        summary.append(f"Effect size (Cohen's d): {effect['effect_size']:.3f} ({effect['magnitude']})")
        
        # Degradation analysis
        degradation = results['degradation_analysis']
        summary.append(f"Meaningful degradations: {degradation['meaningful_count']}/{degradation['total_tested']}")
        
        # Threshold analysis
        threshold = results['threshold_analysis']
        if threshold['threshold_intensity'] is not None:
            summary.append(f"Degradation threshold: {threshold['threshold_intensity']:.1f}")
        
        # Summary statistics
        stats = results['summary_statistics']
        summary.append(f"Baseline AUC: {stats['baseline_auc']:.3f}")
        summary.append(f"Maximum degradation: {stats['max_degradation']:.3f}")
        summary.append(f"Relative degradation: {stats['relative_degradation']:.1f}%")
    
    return "\n".join(summary)

if __name__ == "__main__":
    # Test the statistical utilities
    print("Testing statistical utilities...")
    
    # Create sample results
    sample_results = pd.DataFrame({
        'Task': ['Mortality'] * 5,
        'Debt_Intensity': [0.0, 0.3, 0.5, 0.7, 1.0],
        'AUC': [0.85, 0.83, 0.80, 0.75, 0.70]
    })
    
    # Perform analysis
    analysis = perform_complete_statistical_analysis(sample_results)
    
    # Print summary
    summary = create_statistical_summary(analysis)
    print(summary)
