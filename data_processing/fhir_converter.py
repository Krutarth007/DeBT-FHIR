"""
fhir_converter.py - Converts MIMIC-IV data to FHIR format
"""

import os
import json
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
import math
import logging
import random
from collections import defaultdict, Counter
import numpy as np

class FHIRConverter:
    """Converts MIMIC-IV data to FHIR R4 bundles"""
    
    def __init__(self, base_dir, target_patients=9000, random_seed=42):
        self.base_dir = base_dir
        self.target_patients = target_patients
        self.random_seed = random_seed
        self.csv_map = {}
        
        # Configuration
        self.min_observations = 5
        self.min_admissions = 1
        self.chunk_size = 50000
        
        # Setup directories
        self.out_dir = os.path.join(base_dir, f"mimic_fhir_{target_patients}_output")
        self.subset_dir = os.path.join(base_dir, f"subset_{target_patients}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.logger = logging.getLogger(__name__)
        
    def ensure_dir(self, d):
        """Create directory if it doesn't exist"""
        os.makedirs(d, exist_ok=True)
    
    def find_csvs_in_dir(self, d):
        """Return dict mapping stem -> fullpath for csv files under dir d"""
        out = {}
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".csv"):
                    out[Path(f).stem] = os.path.join(root, f)
        return out
    
    def discover_mimic_csvs(self):
        """Discover CSVs in hosp/ and icu/ under base_dir"""
        csv_map = {}
        for folder in ("hosp", "icu"):
            dirpath = os.path.join(self.base_dir, folder)
            if os.path.isdir(dirpath):
                csv_map.update(self.find_csvs_in_dir(dirpath))
        return csv_map
    
    def find_patients_with_data_smart(self):
        """
        SMART method to find patients with data
        """
        self.logger.info("Finding patients with data using SMART method...")
        
        # Step 1: Get patients with admissions AND reasonable length of stay
        df_admissions = pd.read_csv(self.csv_map["admissions"], low_memory=False, 
                                   usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])
        
        df_admissions['admittime_dt'] = pd.to_datetime(df_admissions['admittime'], errors='coerce')
        df_admissions['dischtime_dt'] = pd.to_datetime(df_admissions['dischtime'], errors='coerce')
        df_admissions['los_days'] = (df_admissions['dischtime_dt'] - df_admissions['admittime_dt']).dt.total_seconds() / 86400
        
        patients_with_good_admissions = set()
        for patient_id, group in df_admissions.groupby('subject_id'):
            if (group['los_days'] >= 1).any():
                patients_with_good_admissions.add(patient_id)
        
        self.logger.info(f"Found {len(patients_with_good_admissions):,} patients with admissions ≥1 day")
        
        # Sample down if too many
        if len(patients_with_good_admissions) > self.target_patients * 3:
            patients_with_good_admissions = set(random.sample(list(patients_with_good_admissions), 
                                                              self.target_patients * 3))
        
        # Step 2-4: Check lab and chart events (simplified for brevity)
        # Full implementation from your original code would go here
        
        # Return selected patients (simplified for example)
        selected_patients = list(patients_with_good_admissions)[:self.target_patients]
        return selected_patients
    
    def extract_mortality_icu_data(self, selected_patients):
        """
        Extract mortality and ICU admission data
        """
        self.logger.info("Extracting mortality and ICU data...")
        
        selected_set = set(selected_patients)
        mortality_data = {}
        icu_data = {}
        detailed_icu_data = {}
        
        try:
            # Load admissions for mortality
            df_admissions = pd.read_csv(self.csv_map["admissions"], low_memory=False,
                                       usecols=['subject_id', 'hadm_id', 'hospital_expire_flag'])
            
            df_admissions_filtered = df_admissions[df_admissions['subject_id'].isin(selected_set)]
            
            for patient_id in selected_set:
                patient_admissions = df_admissions_filtered[df_admissions_filtered['subject_id'] == patient_id]
                if not patient_admissions.empty and (patient_admissions['hospital_expire_flag'] == 1).any():
                    mortality_data[patient_id] = 1
                else:
                    mortality_data[patient_id] = 0
            
            self.logger.info(f"Mortality data extracted for {len(mortality_data)} patients")
            
        except Exception as e:
            self.logger.error(f"Error extracting mortality data: {e}")
            mortality_data = {pid: 0 for pid in selected_patients}
        
        return mortality_data, icu_data, detailed_icu_data
    
    def create_fhir_bundle(self, patient_id, patient_info, admissions, mortality_flag=0):
        """
        Create FHIR bundle for a single patient
        """
        entries = []
        
        # Patient resource
        patient_resource = {
            "resourceType": "Patient",
            "id": f"patient-{patient_id}",
            "identifier": [{"system": "http://mimic.mit.edu/subject", "value": str(patient_id)}],
            "gender": str(patient_info.get('gender', '')).lower(),
        }
        entries.append({"resource": patient_resource})
        
        # Add mortality flag if applicable
        if mortality_flag == 1:
            patient_resource.setdefault('extension', []).append({
                "url": "http://mimic.mit.edu/fhir/StructureDefinition/hospital-mortality-outcome",
                "valueBoolean": True
            })
        
        # Create bundle
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": entries,
            "meta": {
                "source": "MIMIC-IV FHIR Conversion",
                "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"],
                "conversion_time": datetime.now().isoformat()
            }
        }
        
        return bundle
    
    def convert_all_patients(self):
        """
        Main conversion pipeline
        """
        self.logger.info("Starting FHIR conversion...")
        
        # Create output directories
        self.ensure_dir(self.out_dir)
        self.ensure_dir(self.subset_dir)
        
        # Discover CSV files
        self.csv_map = self.discover_mimic_csvs()
        
        # Find patients with data
        selected_patients = self.find_patients_with_data_smart()
        
        # Extract outcomes
        mortality_data, icu_data, detailed_icu_data = self.extract_mortality_icu_data(selected_patients)
        
        # Convert each patient
        for patient_id in selected_patients:
            # Load patient data (simplified)
            # In full implementation, you would load all relevant data
            
            # Create FHIR bundle
            patient_info = {"gender": "unknown"}  # Simplified
            mortality_flag = mortality_data.get(patient_id, 0)
            
            bundle = self.create_fhir_bundle(patient_id, patient_info, [], mortality_flag)
            
            # Save to file
            output_file = os.path.join(self.out_dir, f"patient_{patient_id}.json")
            with open(output_file, 'w') as f:
                json.dump(bundle, f, indent=2)
        
        self.logger.info(f"Conversion complete! Files saved to {self.out_dir}")
        return self.out_dir

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MIMIC-IV data to FHIR format')
    parser.add_argument('--base_dir', required=True, help='Base directory of MIMIC-IV data')
    parser.add_argument('--target_patients', type=int, default=9000, 
                       help='Number of patients to convert')
    
    args = parser.parse_args()
    
    converter = FHIRConverter(args.base_dir, args.target_patients)
    converter.convert_all_patients()

if __name__ == "__main__":
    main()
