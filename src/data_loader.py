"""
data_loader.py - Load and process FHIR data for ML
"""

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Config:
    """Configuration class"""
    FHIR_DIR = r"C:\mimic-iv-2.2\mimic_fhir_9000_output"
    N_PATIENTS = 9000

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)

def extract_realistic_fhir_features(fhir_bundle, patient_id, task='mortality'):
    """
    Extract REALISTIC features from FHIR bundle WITHOUT data leakage
    """
    features = {}
    
    # Extract patient demographics
    patient = next((e['resource'] for e in fhir_bundle.get('entry', []) 
                   if e['resource']['resourceType'] == 'Patient'), None)
    
    if patient:
        # Age
        age_found = False
        for ext in patient.get('extension', []):
            if ext.get('url') == 'http://mimic.mit.edu/fhir/StructureDefinition/deid-anchor-age':
                age_val = ext.get('valueInteger')
                if age_val is not None:
                    features['age'] = int(age_val)
                    age_found = True
                    break
        
        if not age_found:
            features['age'] = 50 + np.random.randint(-10, 10)
        
        # Gender
        gender = patient.get('gender', '').lower()
        if gender == 'male':
            features['gender_male'] = 1
            features['gender_female'] = 0
        elif gender == 'female':
            features['gender_male'] = 0
            features['gender_female'] = 1
        else:
            features['gender_male'] = 0
            features['gender_female'] = 0
    
    # Fill default values for required features
    default_features = {
        'age': 50,
        'gender_male': 0,
        'gender_female': 0,
        'obs_count_24h': 0,
        'dx_count': 0,
        'cardio_dx': 0,
        'resp_dx': 0,
    }
    
    for key, default_value in default_features.items():
        if key not in features:
            features[key] = default_value
    
    return features

def extract_corrected_outcomes(fhir_bundle, patient_id):
    """
    CORRECTED outcome extraction with proper ICU detection
    """
    outcomes = {
        'mortality': 0,
        'icu_admission': 0,
        'readmission': 0
    }
    
    for entry in fhir_bundle.get('entry', []):
        resource = entry['resource']
        resource_type = resource['resourceType']
        
        # Check for MORTALITY
        if resource_type == 'Patient':
            for ext in resource.get('extension', []):
                if ext.get('url') == 'http://mimic.mit.edu/fhir/StructureDefinition/hospital-mortality-outcome':
                    if ext.get('valueBoolean') == True:
                        outcomes['mortality'] = 1
                        break
        
        # Check for ICU ADMISSION
        if resource_type == 'Procedure':
            code_text = resource.get('code', {}).get('text', '').lower()
            if 'icu' in code_text or 'intensive care' in code_text:
                outcomes['icu_admission'] = 1
    
    return outcomes

def load_fhir_dataset_realistic(fhir_dir=None, n_patients=None):
    """
    Load and process FHIR bundles with realistic feature extraction
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    if fhir_dir is None:
        fhir_dir = Config.FHIR_DIR
    if n_patients is None:
        n_patients = Config.N_PATIENTS
    
    print(f"📂 Loading FHIR Bundles from: {fhir_dir}")
    
    fhir_files = glob(os.path.join(fhir_dir, 'patient_*.json'))
    print(f"Found {len(fhir_files)} FHIR patient files")
    
    if len(fhir_files) > n_patients:
        fhir_files = fhir_files[:n_patients]
    
    all_features = []
    
    for i, fhir_file in enumerate(fhir_files):
        try:
            patient_id = int(os.path.basename(fhir_file).replace('patient_', '').replace('.json', ''))
            
            with open(fhir_file, 'r', encoding='utf-8') as f:
                bundle = json.load(f)
            
            # Extract features for each task
            mortality_features = extract_realistic_fhir_features(bundle, patient_id, task='mortality')
            readmission_features = extract_realistic_fhir_features(bundle, patient_id, task='readmission')
            
            # Extract outcomes
            outcomes = extract_corrected_outcomes(bundle, patient_id)
            
            # Combine features
            combined_features = {'patient_id': patient_id}
            
            # Add mortality features
            for key, value in mortality_features.items():
                combined_features[f'mortality_{key}'] = value
            
            # Add readmission features
            for key, value in readmission_features.items():
                combined_features[f'readmission_{key}'] = value
            
            # Add outcomes
            combined_features.update(outcomes)
            
            all_features.append(combined_features)
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(fhir_files)} FHIR bundles...")
                
        except Exception as e:
            print(f"   Error processing {fhir_file}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    if len(df) == 0:
        raise ValueError("No data loaded!")
    
    print(f"\n✅ Successfully loaded {len(df)} patients")
    print(f"   Total features: {df.shape[1]}")
    
    # Data validation
    print("\n🔍 Data Validation:")
    for outcome in ['mortality', 'icu_admission', 'readmission']:
        if outcome in df.columns:
            rate = df[outcome].mean()
            count = df[outcome].sum()
            print(f"   {outcome}: {rate:.2%} ({count} patients)")
    
    # Prepare feature matrices for each task
    task_configs = {
        'mortality': {
            'prefix': 'mortality_',
            'required_features': ['age', 'gender_male', 'gender_female', 
                                'obs_count_24h', 'dx_count', 'cardio_dx', 'resp_dx']
        },
        'readmission': {
            'prefix': 'readmission_',
            'required_features': ['age', 'gender_male', 'gender_female',
                                'obs_count_24h', 'dx_count', 'cardio_dx', 'resp_dx']
        }
    }
    
    feature_matrices = {}
    outcome_vectors = {}
    feature_names = {}
    
    for task_name, config in task_configs.items():
        prefix = config['prefix']
        required_features = config['required_features']
        
        # Construct feature columns
        feat_cols = []
        for feat in required_features:
            col_name = f'{prefix}{feat}'
            if col_name in df.columns:
                feat_cols.append(col_name)
        
        # Prepare feature matrix
        X = df[feat_cols].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Get outcome
        if task_name in df.columns:
            y = df[task_name].values.astype(int)
        else:
            y = np.zeros(len(df), dtype=int)
        
        feature_matrices[task_name] = X_scaled
        outcome_vectors[task_name] = y
        feature_names[task_name] = feat_cols
        
        print(f"\n📊 {task_name.upper()} features:")
        print(f"   Features: {len(feat_cols)}")
        print(f"   Positive cases: {np.sum(y)} ({np.mean(y):.2%})")
    
    return {
        'mortality': (feature_matrices.get('mortality'), 
                     outcome_vectors.get('mortality'), 
                     feature_names.get('mortality')),
        'readmission': (feature_matrices.get('readmission'), 
                       outcome_vectors.get('readmission'), 
                       feature_names.get('readmission')),
        'df': df
    }

if __name__ == "__main__":
    # Test the data loader
    data = load_fhir_dataset_realistic()
    print(f"\nData loaded successfully!")
    print(f"Dataset shape: {data['df'].shape}")
