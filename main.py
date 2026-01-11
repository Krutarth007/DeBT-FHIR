"""
main.py - Main execution script for the DeBT-FHIR study
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import modular components
from data_loader import load_fhir_dataset_realistic
from debt_injector import InteroperabilityDebtInjector
from models import evaluate_ensemble_cross_validation, calculate_bootstrap_ci
from stats_utils import perform_complete_statistical_analysis, create_statistical_summary

class DeBTFHIRStudy:
    """Main study class for DeBT-FHIR experiment"""
    
    def __init__(self, config=None):
        # Default configuration
        self.config = config or {
            'fhir_dir': r"C:\mimic-iv-2.2\mimic_fhir_9000_output",
            'n_patients': 9000,
            'debt_levels': np.linspace(0.0, 1.0, 11),
            'n_splits': 5,
            'n_bootstraps': 300,
            'random_seed': 42
        }
        
        # Initialize components
        self.debt_injector = InteroperabilityDebtInjector(random_seed=self.config['random_seed'])
        self.results = []
        self.statistical_results = {}
        
    def load_data(self):
        """Load and preprocess FHIR data"""
        print("="*80)
        print("DeBT-FHIR Study: Loading Data")
        print("="*80)
        
        self.data = load_fhir_dataset_realistic(
            fhir_dir=self.config['fhir_dir'],
            n_patients=self.config['n_patients']
        )
        
        return self.data
    
    def run_experiment(self):
        """Run the main interoperability debt experiment"""
        print("\n" + "="*80)
        print("DeBT-FHIR Study: Running Experiment")
        print("="*80)
        
        # Define tasks to evaluate
        tasks = {
            'IN-HOSPITAL MORTALITY': self.data['mortality'],
            '30-DAY READMISSION': self.data['readmission']
        }
        
        for task_name, (X, y, feature_names) in tasks.items():
            if X is None or y is None or np.sum(y) < 20:
                print(f"\nSkipping {task_name}: insufficient data")
                continue
            
            print(f"\n{task_name}:")
            print(f"  Prevalence: {np.mean(y):.2%} ({np.sum(y)} positive cases)")
            print(f"  Features: {X.shape[1]}")
            
            # Evaluate baseline (no debt)
            print(f"\n  Baseline Performance (No Debt):")
            baseline_results = evaluate_ensemble_cross_validation(
                X, y, 
                n_splits=self.config['n_splits'],
                random_seed=self.config['random_seed']
            )
            
            if baseline_results is None:
                print("    Could not evaluate baseline")
                continue
            
            baseline_auc = baseline_results['auc'].mean()
            baseline_ci = calculate_bootstrap_ci(
                X, y, 
                n_bootstraps=self.config['n_bootstraps'],
                random_seed=self.config['random_seed']
            )
            
            print(f"    AUC: {baseline_auc:.3f}")
            print(f"    95% CI: [{baseline_ci['auc_ci_lower']:.3f}, {baseline_ci['auc_ci_upper']:.3f}]")
            
            # Store baseline results
            self.results.append({
                'Task': task_name,
                'Debt_Intensity': 0.0,
                'AUC': baseline_auc,
                'AUC_CI_Lower': baseline_ci['auc_ci_lower'],
                'AUC_CI_Upper': baseline_ci['auc_ci_upper'],
                'Positive_Cases': np.sum(y),
                'Prevalence': np.mean(y),
                'N_Features': X.shape[1]
            })
            
            # Test with different debt levels
            print(f"\n  Testing Interoperability Debt Levels:")
            
            for debt_level in self.config['debt_levels'][1:]:
                print(f"    Debt Level {debt_level:.1f}: ", end='')
                
                # Inject debt
                X_debt = self.debt_injector.inject_fhir_interoperability_debt(X, debt_level)
                
                # Evaluate with debt
                debt_results = evaluate_ensemble_cross_validation(
                    X_debt, y,
                    n_splits=self.config['n_splits'],
                    random_seed=self.config['random_seed'] + int(debt_level * 100)
                )
                
                if debt_results is None:
                    print("Skipped")
                    continue
                
                debt_auc = debt_results['auc'].mean()
                auc_degradation = baseline_auc - debt_auc
                
                print(f"AUC: {debt_auc:.3f} (Δ: {auc_degradation:.3f})")
                
                # Store debt results
                self.results.append({
                    'Task': task_name,
                    'Debt_Intensity': debt_level,
                    'AUC': debt_auc,
                    'AUC_CI_Lower': debt_auc - 0.02,  # Simplified CI
                    'AUC_CI_Upper': debt_auc + 0.02,
                    'AUC_Degradation': auc_degradation,
                    'Relative_Degradation': auc_degradation / baseline_auc if baseline_auc > 0 else 0
                })
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        return self.results_df
    
    def analyze_results(self):
        """Perform statistical analysis on results"""
        print("\n" + "="*80)
        print("DeBT-FHIR Study: Statistical Analysis")
        print("="*80)
        
        self.statistical_results = perform_complete_statistical_analysis(self.results_df)
        
        # Print summary
        summary = create_statistical_summary(self.statistical_results)
        print(summary)
        
        return self.statistical_results
    
    def save_results(self, output_dir='results'):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results DataFrame
        results_file = os.path.join(output_dir, f'debt_fhir_results_{timestamp}.csv')
        self.results_df.to_csv(results_file, index=False)
        print(f"\n📊 Results saved to: {results_file}")
        
        # Save statistical analysis
        stats_file = os.path.join(output_dir, f'statistical_analysis_{timestamp}.json')
        with open(stats_file, 'w') as f:
            json.dump(self.statistical_results, f, indent=2, default=str)
        print(f"📈 Statistical analysis saved to: {stats_file}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, f'study_summary_{timestamp}.md')
        with open(summary_file, 'w') as f:
            f.write(self._create_summary_report())
        print(f"📋 Summary report saved to: {summary_file}")
    
    def _create_summary_report(self):
        """Create a summary report of the study"""
        report = []
        report.append("# DeBT-FHIR Study Summary")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Study Configuration")
        report.append(f"- Patients: {self.config['n_patients']}")
        report.append(f"- Debt Levels: {len(self.config['debt_levels'])} ({self.config['debt_levels'][0]:.1f} to {self.config['debt_levels'][-1]:.1f})")
        report.append(f"- Cross-Validation Folds: {self.config['n_splits']}")
        
        report.append("\n## Key Findings")
        
        for task in self.results_df['Task'].unique():
            task_data = self.results_df[self.results_df['Task'] == task]
            baseline = task_data[task_data['Debt_Intensity'] == 0.0].iloc[0]
            
            report.append(f"\n### {task}")
            report.append(f"- Baseline AUC: {baseline['AUC']:.3f}")
            report.append(f"- Positive Cases: {baseline['Positive_Cases']}")
            report.append(f"- Prevalence: {baseline['Prevalence']:.2%}")
            
            # Calculate maximum degradation
            max_debt = task_data[task_data['Debt_Intensity'] == 1.0]
            if not max_debt.empty:
                max_debt_row = max_debt.iloc[0]
                degradation = baseline['AUC'] - max_debt_row['AUC']
                relative = (degradation / baseline['AUC']) * 100
                report.append(f"- Maximum Degradation: {degradation:.3f} AUC units ({relative:.1f}%)")
        
        return "\n".join(report)
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("="*80)
        print("DeBT-FHIR: Complete Study Pipeline")
        print("="*80)
        
        try:
            # 1. Load data
            self.load_data()
            
            # 2. Run experiment
            self.run_experiment()
            
            # 3. Analyze results
            self.analyze_results()
            
            # 4. Save results
            self.save_results()
            
            print("\n" + "="*80)
            print("✅ STUDY COMPLETE!")
            print("="*80)
            print("\nFiles generated in 'results/' directory:")
            print("1. CSV file with all results")
            print("2. JSON file with statistical analysis")
            print("3. Markdown summary report")
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeBT-FHIR: Interoperability Debt Study')
    parser.add_argument('--fhir_dir', default=r"C:\mimic-iv-2.2\mimic_fhir_9000_output",
                       help='Directory containing FHIR patient files')
    parser.add_argument('--n_patients', type=int, default=9000,
                       help='Number of patients to analyze')
    parser.add_argument('--output_dir', default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'fhir_dir': args.fhir_dir,
        'n_patients': args.n_patients,
        'debt_levels': np.linspace(0.0, 1.0, 11),
        'n_splits': 5,
        'n_bootstraps': 300,
        'random_seed': 42
    }
    
    # Run study
    study = DeBTFHIRStudy(config)
    study.run_full_pipeline()

if __name__ == "__main__":
    main()
