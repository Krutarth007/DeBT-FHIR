# DeBT-FHIR: Interoperability Debt Simulation on MIMIC-IV FHIR Data


This repository contains the complete code and data processing pipeline for the study:

> **"The Impact of FHIR Interoperability Debt on Clinical AI Reliability: A Stress-Testing Framework and Simulation Study"**  
> *Health Informatics Journal Submission*

## 📋 Study Overview

This research quantifies how **FHIR (Fast Healthcare Interoperability Resources) interoperability debt**—accumulated data quality issues during health data exchange—impacts the reliability of clinical artificial intelligence (AI) models. Using a cohort of 9,000 patients from MIMIC-IV converted to FHIR format, we simulate real-world data exchange degradation and evaluate its effect on two high-stakes prediction tasks: **in-hospital mortality** and **30-day readmission**.

### Key Findings:
- **Critical Threshold:** 70% interoperability debt intensity
- **Maximum Degradation:** 2.9% AUC reduction (mortality), 7.9% AUC reduction (readmission)
- **Strong Correlation:** Spearman ρ = -0.847 for mortality (p < 0.001)

## 🚀 Quick Start (For Reviewers)

### Prerequisites
- Python 3.9 or higher
- Access to MIMIC-IV v2.2 (requires [PhysioNet credentialed access](https://physionet.org/content/mimiciv/))
- ~8GB RAM minimum

### Installation
```bash
# Clone repository
git clone https://github.com/krutarth007/DeBT-FHIR.git
cd DeBT-FHIR
```
## Install dependencies
pip install -r requirements.txt

## Complete Reproduction (Two Steps)
### Step 1: Convert MIMIC-IV to FHIR
```bash
# Run the complete FHIR converter (takes 30-60 minutes)
python main_fhir_convert.py --base_dir /path/to/your/mimic-iv-2.2
```
### Step 2: Run the Main Study
```bash
# Run the complete study pipeline (takes 2-3 hours)
python main_pipeline_combined.py
```
### Expected Outputs:

outputs/results_*.csv - Performance metrics across debt levels  
outputs/figures/ - figures (PNG and PDF)  
outputs/summary_*.json - Statistical analysis results  


## 📁 Repository Structure
```
DeBT-FHIR/
├── main_fhir_convert.py              # COMPLETE FHIR converter (9000 patients)
├── main_pipeline_combined.py         # COMPLETE study pipeline
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
├── data_processing/                  # Modular FHIR conversion
│   └── fhir_converter.py            # Modular converter version
│
├── src/                              # Modular study components
│   ├── data_loader.py               # FHIR feature extraction
│   ├── debt_injector.py             # Interoperability debt simulation
│   ├── models.py                    # ML ensemble (Logistic Regression + Random Forest)
│   └── stats_utils.py               # Statistical analysis utilities
│
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── MIMIC_IV_to_FHIR_Converter_9000.ipynb          # Interactive FHIR conversion
│   └── DeBT_FHIR_Interoperability_Debt_Study.ipynb    # Interactive study pipeline
│
└── outputs/                          # Generated results (gitignored)
    ├── figures/                      # Publication figures
    ├── results/                      # CSV results files**
```

## 🔧 Detailed Usage
## Option A: Direct Script Execution (Recommended for Reproduction)
### 1. Convert MIMIC-IV to FHIR (adjust paths as needed)
```bash
python main_fhir_convert.py --base_dir "C:\mimic-iv-2.2" --target_patients 9000
```
### 2. Run the complete study
```bash
python main_pipeline_combined.py --fhir_dir "C:\mimic-iv-2.2\mimic_fhir_9000_output" --n_patients 9000
```
## Option B: Modular Execution
### Using the modular codebase (for development/extensions)
```bash
python -m src.data_loader            # Test data loading
python -m src.debt_injector          # Test debt injection
```
## Option C: Jupyter Notebooks
```bash
# Launch Jupyter and run notebooks interactively
Then open: 
# 1. MIMIC_IV_to_FHIR_Converter_9000.ipynb
# 2. DeBT_FHIR_Interoperability_Debt_Study.ipynb
```

## 📊 Output Files  
After running main_pipeline_combined.py, you'll find:
## Results Files:
#### outputs/results_YYYYMMDD_HHMMSS.csv - Complete performance metrics  
Columns: Task, Debt_Intensity, AUC, AUC_CI_Lower, AUC_CI_Upper, PR_AUC, Brier, etc.
#### outputs/statistical_analysis_YYYYMMDD_HHMMSS.json - Statistical tests
Spearman correlation, linear regression, effect sizes (Cohen's d)

## Figures (Publication-Ready):
outputs/figures/Figure1_Performance_Degradation.png/pdf - AUC vs. debt intensity  
outputs/figures/Figure2_Statistical_Significance.png/pdf - Effect size visualization

## Study Summary:
outputs/study_summary_YYYYMMDD_HHMMSS.md - Human-readable summary  
outputs/HIJ_Submission_Summary_*.json - Manuscript-ready summary  

## ⚙️ Configuration
### Path Configuration (Edit if needed):
**MIMIC-IV Directory:** Update in main_fhir_convert.py or use --base_dir argument  
**FHIR Output Directory:** Default: C:\mimic-iv-2.2\mimic_fhir_9000_output  
**Study Parameters:** Debt levels, ML settings in main_pipeline_combined.py  

## Key Study Parameters:
**Patients**: 9,000 adult patients (age ≥18)  
**Debt Levels**: 11 levels (0.0, 0.1, ..., 1.0)  
**ML Models**: Ensemble of Logistic Regression + Random Forest  
**Cross-Validation**: 5-fold stratified  
**Bootstraps**: 500 iterations for confidence intervals


## 🧪 Validation
To verify successful reproduction:  
```bash
# Check that key results match paper claims
python -c "
import pandas as pd, glob, json
results = glob.glob('outputs/results_*.csv')[0]
df = pd.read_csv(results)
baseline = df[df['Debt_Intensity']==0]
print('✓ Baseline AUC - Mortality:', baseline[baseline['Task']=='IN-HOSPITAL MORTALITY']['AUC'].iloc[0])
print('✓ Baseline AUC - Readmission:', baseline[baseline['Task']=='30-DAY READMISSION']['AUC'].iloc[0])
"
```

**Expected Output:**
```
✓ Baseline AUC - Mortality: 0.821
✓ Baseline AUC - Readmission: 0.892
```

## 🔬 Methodology Details
### 1. Data Processing
Source: MIMIC-IV v2.2 (9,000 patients)  
Conversion: Custom Python pipeline to FHIR R4 bundles  
Feature Extraction: Demographics, lab values, diagnoses, observation counts
Outcomes: In-hospital mortality, 30-day readmission (validated extraction)

### 2. Interoperability Debt Simulation
**Seven real-world debt types implemented:  
Random missingness  
Systematic missing data  
Measurement noise  
Severe data loss  
Outlier corruption  
Precision reduction (feature collapse)  
Noisy imputation**

### 3. Machine Learning Pipeline
**Models: Ensemble (Logistic Regression + Random Forest)  
Validation: Stratified 5-fold cross-validation  
Metrics: AUC (primary), PR-AUC, Brier score, sensitivity, specificity  
Uncertainty: 500 bootstrap iterations for confidence intervals**

### 4. Statistical Analysis
**Correlation: Spearman's rank correlation  
Regression: Linear modeling of AUC vs. debt intensity  
Effect Size: Cohen's d for clinical significance  
Threshold Detection: Clinical meaningful degradation (>0.02 AUC drop)**

## 🐛 Troubleshooting
### Common Issues:
#### "MIMIC-IV files not found"
Ensure you have downloaded MIMIC-IV v2.2 from PhysioNet  
Update the --base_dir argument to point to your MIMIC-IV directory  

#### "Memory error"
The study requires ~8GB RAM for 9,000 patients  
Reduce patient count: --n_patients 5000  
Close other memory-intensive applications  

#### "Dependency installation fails"

Use Python 3.9 or higher  
Try: pip install --upgrade pip first  
Install packages individually if needed  

#### "Long execution time"

Full pipeline: 3-4 hours (FHIR conversion + study)  
Use --target_patients 1000 for faster testing  

## Getting Help:  
Check outputs/logs/ for detailed error logs  
Ensure all file paths are correct  
Verify MIMIC-IV data integrity  

## 📝 Citation
If you use this code or reference this study, please cite:  

```
bibtex
@article{patel2024debtfhirsensitivity,
  title={Quantifying the Impact of FHIR Interoperability Debt on Clinical Machine Learning Performance: A Large-Scale Simulation Study Using MIMIC-IV},
  author={Patel, Krutarth},
  journal={Journal of the American Medical Informatics Association},
  year={2024},
  note={Manuscript submitted for publication}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.  

## 🙏 Acknowledgments
**MIMIC-IV Database**: Massachusetts Institute of Technology Laboratory for Computational Physiology

**PhysioNet**: For hosting and maintaining MIMIC-IV

**HL7 FHIR Community**: For the interoperability standards

**HIJ Reviewers**: For their valuable feedback

## 📧 Contact
For questions about this study or code:  
**Author**: Krutarth Patel  
**Email**: krutarthpatel1997@gmail.com   
**GitHub Issues**: Create an issue  






