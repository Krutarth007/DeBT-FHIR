#!/usr/bin/env python3
"""
mimic_to_fhir_9000_FIXED_WITH_MORTALITY.py

FIXED ULTRA FAST converter - actually finds patients with data!
Now includes proper mortality and ICU data extraction.
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

# ---------------- USER CONFIG ----------------
BASE_DIR = r"C:\mimic-iv-2.2"   # <- CHANGE this to your local MIMIC folder root

# === TARGET: 9000 PATIENTS ===
TARGET_PATIENTS = 9000              # UPDATED from 8500 to 9000
MIN_OBSERVATIONS = 5               # LOWERED from 10 to 5 for more patients
MIN_ADMISSIONS = 1                  # Must have at least 1 admission
# ============================================

# Dynamic output directory names
OUT_DIR = os.path.join(BASE_DIR, f"mimic_fhir_{TARGET_PATIENTS}_output")
SUBSET_DIR = os.path.join(BASE_DIR, f"subset_{TARGET_PATIENTS}")

RANDOM_SEED = 42
N_CLIENTS = 50
CHUNK_SIZE = 50000  # For streaming large files
SYNTHETIC_ABSOLUTE = False
SYNTHETIC_EPOCH = datetime(2000, 1, 1)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def find_csvs_in_dir(d: str):
    """Return dict mapping stem -> fullpath for csv files under dir d."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".csv"):
                out[Path(f).stem] = os.path.join(root, f)
    return out

def discover_mimic_csvs(base_dir: str):
    """Discover CSVs in hosp/ and icu/ under base_dir."""
    csv_map = {}
    for folder in ("hosp", "icu"):
        dirpath = os.path.join(base_dir, folder)
        if os.path.isdir(dirpath):
            csv_map.update(find_csvs_in_dir(dirpath))
    return csv_map

def find_patients_with_data_smart(csv_map, target_count=9000):
    """
    SMART method to find patients with data:
    1. Get ALL patients with admissions
    2. Sample from admissions to find patients with hospital stays
    3. Check those patients in lab/chart events with MORE sampling
    """
    logging.info("Finding patients with data using SMART method...")
    
    # Step 1: Get patients with admissions AND reasonable length of stay
    logging.info("  Step 1: Finding patients with meaningful admissions...")
    df_admissions = pd.read_csv(csv_map["admissions"], low_memory=False, 
                               usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])
    
    # Calculate length of stay for each admission
    df_admissions['admittime_dt'] = pd.to_datetime(df_admissions['admittime'], errors='coerce')
    df_admissions['dischtime_dt'] = pd.to_datetime(df_admissions['dischtime'], errors='coerce')
    df_admissions['los_days'] = (df_admissions['dischtime_dt'] - df_admissions['admittime_dt']).dt.total_seconds() / 86400
    
    # Filter patients with at least one admission of 1+ days (more likely to have data)
    patients_with_good_admissions = set()
    for patient_id, group in df_admissions.groupby('subject_id'):
        if (group['los_days'] >= 1).any():  # At least 1 day stay
            patients_with_good_admissions.add(patient_id)
    
    logging.info(f"    Found {len(patients_with_good_admissions):,} patients with admissions ≥1 day")
    
    # If we have too many, sample down
    if len(patients_with_good_admissions) > target_count * 3:
        patients_with_good_admissions = set(random.sample(list(patients_with_good_admissions), target_count * 3))
        logging.info(f"    Sampled down to {len(patients_with_good_admissions):,} patients")
    
    # Step 2: Check labevents for these patients - READ MORE CHUNKS!
    logging.info("  Step 2: Checking lab events (reading 20 chunks)...")
    patient_lab_counts = Counter()
    candidate_set = patients_with_good_admissions
    
    try:
        chunks_processed = 0
        total_rows = 0
        for chunk in pd.read_csv(csv_map["labevents"], chunksize=CHUNK_SIZE, low_memory=False,
                                usecols=['subject_id', 'valuenum']):
            chunks_processed += 1
            total_rows += len(chunk)
            
            # Filter for our candidate patients
            chunk_filtered = chunk[chunk['subject_id'].isin(candidate_set)]
            
            # Count patients with numeric values
            if not chunk_filtered.empty:
                # Group by patient and count non-null values
                patient_counts = chunk_filtered.groupby('subject_id')['valuenum'].apply(
                    lambda x: x.notna().sum()
                )
                for patient_id, count in patient_counts.items():
                    patient_lab_counts[patient_id] += count
            
            # Log progress
            if chunks_processed % 5 == 0:
                logging.info(f"      Processed {chunks_processed} chunks, {total_rows:,} total rows")
            
            # READ MORE CHUNKS - 20 instead of 5!
            if chunks_processed >= 20:
                break
                
    except Exception as e:
        logging.error(f"    Error reading labevents: {e}")
    
    # Step 3: Check chartevents for these patients - READ MORE CHUNKS!
    logging.info("  Step 3: Checking chart events (reading 20 chunks)...")
    patient_chart_counts = Counter()
    
    try:
        chunks_processed = 0
        total_rows = 0
        for chunk in pd.read_csv(csv_map["chartevents"], chunksize=CHUNK_SIZE, low_memory=False,
                                usecols=['subject_id', 'valuenum']):
            chunks_processed += 1
            total_rows += len(chunk)
            
            chunk_filtered = chunk[chunk['subject_id'].isin(candidate_set)]
            
            if not chunk_filtered.empty:
                patient_counts = chunk_filtered.groupby('subject_id')['valuenum'].apply(
                    lambda x: x.notna().sum()
                )
                for patient_id, count in patient_counts.items():
                    patient_chart_counts[patient_id] += count
            
            if chunks_processed % 5 == 0:
                logging.info(f"      Processed {chunks_processed} chunks, {total_rows:,} total rows")
            
            # READ MORE CHUNKS!
            if chunks_processed >= 20:
                break
                
    except Exception as e:
        logging.error(f"    Error reading chartevents: {e}")
    
    # Step 4: Combine counts and select patients
    logging.info("  Step 4: Selecting final patients...")
    
    patient_total_counts = []
    for patient_id in candidate_set:
        total_obs = patient_lab_counts.get(patient_id, 0) + patient_chart_counts.get(patient_id, 0)
        if total_obs >= MIN_OBSERVATIONS:
            patient_total_counts.append((patient_id, total_obs))
    
    # Sort by observation count (highest first)
    patient_total_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Take the best patients
    selected_patients = [pid for pid, count in patient_total_counts[:target_count]]
    
    logging.info(f"    Found {len(patient_total_counts)} patients with ≥{MIN_OBSERVATIONS} observations")
    logging.info(f"    Selected {len(selected_patients)} patients")
    
    # If still not enough, be even more lenient
    if len(selected_patients) < target_count:
        logging.warning(f"Only found {len(selected_patients)} patients, need {target_count}")
        logging.warning("Being more lenient: including patients with ANY observations...")
        
        # Add patients with any observations
        remaining_needed = target_count - len(selected_patients)
        additional_candidates = []
        
        for patient_id in candidate_set:
            if patient_id not in selected_patients:
                total_obs = patient_lab_counts.get(patient_id, 0) + patient_chart_counts.get(patient_id, 0)
                if total_obs > 0:
                    additional_candidates.append((patient_id, total_obs))
        
        additional_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_patients.extend([pid for pid, count in additional_candidates[:remaining_needed]])
        
        logging.info(f"    Now have {len(selected_patients)} patients")
    
    # Get statistics
    if selected_patients:
        stats_sample = random.sample(selected_patients, min(50, len(selected_patients)))
        sample_stats = []
        for pid in stats_sample:
            total_obs = patient_lab_counts.get(pid, 0) + patient_chart_counts.get(pid, 0)
            sample_stats.append(total_obs)
        
        logging.info(f"  Sample statistics (n={len(sample_stats)}):")
        logging.info(f"    Avg observations in sample: {np.mean(sample_stats):.1f}")
        logging.info(f"    Min observations: {min(sample_stats)}")
        logging.info(f"    Max observations: {max(sample_stats)}")
        logging.info(f"    Patients with 10+ obs: {sum(1 for x in sample_stats if x >= 10)}")
    
    return selected_patients

def to_iso(dt):
    """Convert pandas timestamp or string to ISO str; return None on failure."""
    if pd.isnull(dt):
        return None
    try:
        ts = pd.to_datetime(dt, errors='coerce')
        if pd.isnull(ts):
            return None
        return ts.isoformat()
    except Exception:
        return None

def compute_offset_days(ref_ts, ts):
    """Return integer days offset (ts - ref_ts). If ts invalid return None."""
    if ref_ts is None or ts is None:
        return None
    try:
        a = pd.to_datetime(ref_ts, errors='coerce')
        b = pd.to_datetime(ts, errors='coerce')
        if pd.isnull(a) or pd.isnull(b):
            return None
        delta = b - a
        # Round to nearest integer day (floor)
        return int(math.floor(delta.total_seconds() / 86400.0))
    except Exception:
        return None

def create_filtered_subset(selected_patients, csv_map, subset_dir):
    """Create filtered CSV files for selected patients."""
    logging.info("Creating filtered subset CSVs...")
    
    selected_set = set(selected_patients)
    
    # Filter each table
    tables_to_process = [
        'patients', 'admissions', 'icustays', 'diagnoses_icd',
        'procedures_icd', 'prescriptions', 'labevents', 'chartevents',
        'microbiologyevents', 'services'
    ]
    
    for table in tables_to_process:
        if table not in csv_map:
            logging.warning(f"  Skipping {table} - not found")
            continue
            
        input_path = csv_map[table]
        output_path = os.path.join(subset_dir, f"{table}.csv")
        
        logging.info(f"  Filtering {table}...")
        
        try:
            # For patients table, just get our selected patients
            if table == 'patients':
                df_all = pd.read_csv(input_path, low_memory=False)
                df_filtered = df_all[df_all['subject_id'].isin(selected_set)]
                df_filtered.to_csv(output_path, index=False)
                logging.info(f"    Kept {len(df_filtered)} patients")
                continue
            
            # For other tables, check if they have subject_id
            df_sample = pd.read_csv(input_path, nrows=1, low_memory=False)
            has_subject_id = 'subject_id' in df_sample.columns
            
            if not has_subject_id:
                # Table doesn't have subject_id - copy entire file
                import shutil
                shutil.copy2(input_path, output_path)
                logging.info(f"    Copied entire file (no subject_id)")
                continue
            
            # For large tables, use chunked processing
            if table in ['labevents', 'chartevents']:
                first_chunk = True
                total_kept = 0
                
                for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False):
                    chunk_filtered = chunk[chunk['subject_id'].isin(selected_set)]
                    
                    if not chunk_filtered.empty:
                        if first_chunk:
                            chunk_filtered.to_csv(output_path, index=False, mode='w')
                            first_chunk = False
                        else:
                            chunk_filtered.to_csv(output_path, index=False, mode='a', header=False)
                        
                        total_kept += len(chunk_filtered)
                
                logging.info(f"    Kept {total_kept:,} rows")
            else:
                # For smaller tables, read all at once
                df_all = pd.read_csv(input_path, low_memory=False)
                df_filtered = df_all[df_all['subject_id'].isin(selected_set)]
                df_filtered.to_csv(output_path, index=False)
                logging.info(f"    Kept {len(df_filtered):,} rows")
                
        except Exception as e:
            logging.error(f"    Error filtering {table}: {e}")
            # Create empty file as placeholder
            pd.DataFrame().to_csv(output_path, index=False)

# Updated function to extract mortality and ICU data with proper ICU extraction
def extract_mortality_icu_data(csv_map, selected_patients):
    """
    Extract mortality and ICU admission data for selected patients.
    Returns dictionaries with patient_id as key.
    """
    logging.info("Extracting mortality and ICU data...")
    
    selected_set = set(selected_patients)
    mortality_data = {}
    icu_data = {}
    detailed_icu_data = {}  # Store detailed ICU data
    
    try:
        # Load admissions data for mortality
        df_admissions = pd.read_csv(csv_map["admissions"], low_memory=False,
                                   usecols=['subject_id', 'hadm_id', 'hospital_expire_flag'])
        
        # Filter for selected patients
        df_admissions_filtered = df_admissions[df_admissions['subject_id'].isin(selected_set)]
        
        # Extract mortality data - check all admissions for each patient
        for patient_id in selected_set:
            patient_admissions = df_admissions_filtered[df_admissions_filtered['subject_id'] == patient_id]
            # If patient has any admission with hospital_expire_flag = 1, they died in hospital
            if not patient_admissions.empty and (patient_admissions['hospital_expire_flag'] == 1).any():
                mortality_data[patient_id] = 1
            else:
                mortality_data[patient_id] = 0
        
        logging.info(f"  Mortality data extracted for {len(mortality_data)} patients")
        if mortality_data:
            mortality_count = sum(mortality_data.values())
            mortality_rate = mortality_count / len(mortality_data) * 100
            logging.info(f"  Raw mortality rate: {mortality_rate:.2f}% ({mortality_count} deaths)")
    except Exception as e:
        logging.error(f"  Error extracting mortality data: {e}")
        # Initialize with zeros if extraction fails
        mortality_data = {pid: 0 for pid in selected_patients}
    
    try:
        # Load ICU stays data - read ALL columns to get proper ICU data
        df_icustays = pd.read_csv(csv_map["icustays"], low_memory=False,
                                 usecols=['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 
                                         'intime', 'outtime', 'los'])
        
        # Filter for selected patients
        df_icustays_filtered = df_icustays[df_icustays['subject_id'].isin(selected_set)]
        
        # Create detailed ICU data structure for each patient
        for _, row in df_icustays_filtered.iterrows():
            patient_id = row['subject_id']
            if patient_id not in detailed_icu_data:
                detailed_icu_data[patient_id] = []
            
            detailed_icu_data[patient_id].append({
                'hadm_id': row['hadm_id'],
                'stay_id': row['stay_id'],
                'first_careunit': row['first_careunit'],
                'intime': row['intime'],
                'outtime': row['outtime'],
                'los': row['los'] if pd.notnull(row['los']) else None
            })
        
        # Create binary ICU admission flag
        icu_data = {pid: 1 if pid in detailed_icu_data else 0 for pid in selected_patients}
        
        logging.info(f"  ICU data extracted for {len(icu_data)} patients")
        if icu_data:
            icu_count = sum(icu_data.values())
            icu_rate = icu_count / len(icu_data) * 100
            total_icu_stays = sum(len(stays) for stays in detailed_icu_data.values())
            logging.info(f"  Raw ICU admission rate: {icu_rate:.2f}% ({icu_count} ICU patients)")
            logging.info(f"  Total ICU stays: {total_icu_stays}")
            
            # Log sample of ICU details
            if detailed_icu_data:
                sample_patients = list(detailed_icu_data.keys())[:3]
                for pid in sample_patients:
                    stays = detailed_icu_data[pid]
                    logging.info(f"  Sample patient {pid}: {len(stays)} ICU stays, units: {[s['first_careunit'] for s in stays]}")
        
        return mortality_data, icu_data, detailed_icu_data
        
    except Exception as e:
        logging.error(f"  Error extracting ICU data: {e}")
        # Initialize with zeros if extraction fails
        icu_data = {pid: 0 for pid in selected_patients}
        detailed_icu_data = {}
        return mortality_data, icu_data, detailed_icu_data

# Main pipeline
def main():
    logging.info("="*60)
    logging.info("MIMIC-IV to FHIR Converter - FIXED VERSION WITH PROPER ICU DATA")
    logging.info(f"Target: {TARGET_PATIENTS} patients")
    logging.info(f"Strategy: Find patients with hospital stays ≥1 day")
    logging.info("="*60)
    
    start_time = datetime.now()
    
    # Step 1: Discover files
    logging.info("Step 1: Discovering CSV files...")
    csv_map = discover_mimic_csvs(BASE_DIR)
    if "patients" not in csv_map:
        logging.error("Could not find patients.csv!")
        return
    
    logging.info(f"Found {len(csv_map)} CSV files")
    
    # Step 2: Create output directories
    ensure_dir(SUBSET_DIR)
    ensure_dir(OUT_DIR)
    logging.info(f"Output will be saved to: {OUT_DIR}")
    
    # Step 3: Find patients with data
    logging.info("\nStep 2: Finding patients with data (reading 20+ chunks)...")
    selected_patient_ids = find_patients_with_data_smart(csv_map, TARGET_PATIENTS)
    
    if len(selected_patient_ids) < TARGET_PATIENTS * 0.5:
        logging.error(f"Only found {len(selected_patient_ids)} qualified patients.")
        logging.error("Trying alternative approach...")
        
        # Alternative: Just take random patients with admissions
        df_admissions = pd.read_csv(csv_map["admissions"], usecols=['subject_id'], low_memory=False)
        all_patients_with_admissions = df_admissions['subject_id'].unique().tolist()
        
        if len(all_patients_with_admissions) >= TARGET_PATIENTS:
            selected_patient_ids = random.sample(all_patients_with_admissions, TARGET_PATIENTS)
            logging.info(f"Selected {len(selected_patient_ids)} random patients with admissions")
        else:
            logging.error(f"Not enough patients with admissions. Found: {len(all_patients_with_admissions)}")
            return
    
    logging.info(f"Selected {len(selected_patient_ids)} patients")
    
    # Step 4: Extract mortality and ICU data with proper ICU extraction
    logging.info("\nStep 3: Extracting mortality and ICU data...")
    mortality_data, icu_data, detailed_icu_data = extract_mortality_icu_data(csv_map, selected_patient_ids)
    
    # Step 5: Create filtered subset
    logging.info("\nStep 4: Creating filtered subset CSVs...")
    create_filtered_subset(selected_patient_ids, csv_map, SUBSET_DIR)
    
    # Step 6: Load subset data
    logging.info("\nStep 5: Loading subset data for FHIR conversion...")
    
    def load_csv_safe(table_name):
        path = os.path.join(SUBSET_DIR, f"{table_name}.csv")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception as e:
                logging.warning(f"  Could not load {table_name}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    # Load the data
    df_patients = load_csv_safe("patients")
    df_admissions = load_csv_safe("admissions")
    df_labevents = load_csv_safe("labevents")
    df_chartevents = load_csv_safe("chartevents")
    df_diagnoses = load_csv_safe("diagnoses_icd")
    df_procedures = load_csv_safe("procedures_icd")
    df_prescriptions = load_csv_safe("prescriptions")
    df_icustays = load_csv_safe("icustays")
    
    logging.info(f"Loaded: {len(df_patients)} patients, {len(df_labevents):,} labs, {len(df_chartevents):,} charts")
    
    # Step 7: Create FHIR Bundles
    logging.info("\nStep 6: Creating FHIR Bundles...")
    
    total_converted = 0
    patient_stats = []
    
    # Utility function for resource IDs
    def make_id(prefix, *parts):
        parts_str = [str(p) for p in parts if p is not None and str(p).strip() != '']
        return f"{prefix}-{'-'.join(parts_str)}"
    
    for idx, patient_id in enumerate(selected_patient_ids, 1):
        try:
            # Get patient info
            patient_rows = df_patients[df_patients['subject_id'] == patient_id]
            if patient_rows.empty:
                continue
                
            patient_info = patient_rows.iloc[0].to_dict()
            
            # Find reference time (earliest admission)
            patient_admissions = df_admissions[df_admissions['subject_id'] == patient_id]
            t0 = None
            if not patient_admissions.empty and 'admittime' in patient_admissions.columns:
                admittimes = patient_admissions['admittime'].dropna().tolist()
                if admittimes:
                    try:
                        t0_dates = pd.to_datetime(admittimes, errors='coerce')
                        valid_dates = t0_dates[~t0_dates.isna()]
                        if not valid_dates.empty:
                            t0 = valid_dates.min().isoformat()
                    except:
                        t0 = None
            
            # Start building FHIR Bundle
            entries = []
            
            # 1. Patient resource
            patient_resource = {
                "resourceType": "Patient",
                "id": make_id("patient", patient_id),
                "identifier": [{"system": "http://mimic.mit.edu/subject", "value": str(patient_id)}],
                "gender": str(patient_info.get('gender', '')).lower(),
                "extension": [
                    {
                        "url": "http://mimic.mit.edu/fhir/StructureDefinition/deid-anchor-year",
                        "valueString": str(patient_info.get('anchor_year', ''))
                    },
                    {
                        "url": "http://mimic.mit.edu/fhir/StructureDefinition/deid-anchor-age",
                        "valueInteger": int(patient_info.get('anchor_age')) 
                        if pd.notnull(patient_info.get('anchor_age')) else None
                    }
                ]
            }
            entries.append({"resource": patient_resource})
            
            # 2. Admissions as Encounters
            admission_count = 0
            for _, adm in patient_admissions.iterrows():
                admission_count += 1
                hadm_id = adm.get('hadm_id')
                if pd.isna(hadm_id):
                    continue
                    
                enc_resource = {
                    "resourceType": "Encounter",
                    "id": make_id("enc", patient_id, hadm_id),
                    "subject": {"reference": f"Patient/{patient_resource['id']}"},
                    "identifier": [{"system": "http://mimic.mit.edu/hadm", "value": str(hadm_id)}],
                    "period": {
                        "start": to_iso(adm.get('admittime')),
                        "end": to_iso(adm.get('dischtime'))
                    },
                    "hospitalization": {
                        "dischargeDisposition": {
                            "text": str(adm.get('discharge_location', ''))
                        }
                    } if pd.notnull(adm.get('discharge_location')) else {}
                }
                
                # Add hospital mortality flag if available
                if 'hospital_expire_flag' in adm and pd.notnull(adm['hospital_expire_flag']):
                    expire_flag = int(adm['hospital_expire_flag'])
                    if expire_flag == 1:
                        enc_resource.setdefault('extension', []).append({
                            "url": "http://mimic.mit.edu/fhir/StructureDefinition/hospital-mortality",
                            "valueBoolean": True
                        })
                        # Also add outcome as a condition
                        death_condition = {
                            "resourceType": "Condition",
                            "id": make_id("death", patient_id, hadm_id),
                            "subject": {"reference": f"Patient/{patient_resource['id']}"},
                            "code": {
                                "coding": [{
                                    "system": "http://snomed.info/sct",
                                    "code": "419099009",
                                    "display": "Dead"
                                }]
                            },
                            "clinicalStatus": {
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "inactive"
                                }]
                            },
                            "verificationStatus": {
                                "coding": [{
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                    "code": "confirmed"
                                }]
                            }
                        }
                        entries.append({"resource": death_condition})
                
                # Add offset extensions
                if t0:
                    start_offset = compute_offset_days(t0, adm.get('admittime'))
                    end_offset = compute_offset_days(t0, adm.get('dischtime'))
                    
                    if start_offset is not None:
                        enc_resource.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days-start",
                            "valueInteger": int(start_offset)
                        })
                    if end_offset is not None:
                        enc_resource.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days-end",
                            "valueInteger": int(end_offset)
                        })
                
                entries.append({"resource": enc_resource})
            
            # 3. ICU stays as proper resources with detailed data
            icu_count = 0
            # Get detailed ICU data for this patient if available
            patient_detailed_icu = detailed_icu_data.get(patient_id, [])
            
            for icu_idx, icu_stay in enumerate(patient_detailed_icu, 1):
                icu_count += 1
                hadm_id = icu_stay['hadm_id']
                stay_id = icu_stay['stay_id']
                
                # Create ICU Procedure resource (representing ICU stay)
                icu_procedure = {
                    "resourceType": "Procedure",
                    "id": make_id("icu", patient_id, stay_id),
                    "subject": {"reference": f"Patient/{patient_resource['id']}"},
                    "status": "completed",
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "308335008",  # Patient encounter procedure
                            "display": "Patient encounter procedure"
                        }],
                        "text": f"ICU Stay in {icu_stay['first_careunit']}"
                    },
                    "performedPeriod": {
                        "start": to_iso(icu_stay['intime']),
                        "end": to_iso(icu_stay['outtime'])
                    },
                    "extension": [
                        {
                            "url": "http://mimic.mit.edu/fhir/StructureDefinition/icu-stay-details",
                            "extension": [
                                {
                                    "url": "stayId",
                                    "valueString": str(stay_id)
                                },
                                {
                                    "url": "icuUnit",
                                    "valueString": str(icu_stay['first_careunit'])
                                }
                            ]
                        }
                    ]
                }
                
                # Add LOS if available
                if icu_stay['los'] is not None:
                    try:
                        icu_procedure['extension'][0]['extension'].append({
                            "url": "lengthOfStay",
                            "valueDecimal": float(icu_stay['los'])
                        })
                    except:
                        pass
                
                # Add offset extensions if reference time available
                if t0:
                    start_offset = compute_offset_days(t0, icu_stay['intime'])
                    end_offset = compute_offset_days(t0, icu_stay['outtime'])
                    
                    if start_offset is not None:
                        icu_procedure.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days-start",
                            "valueInteger": int(start_offset)
                        })
                    if end_offset is not None:
                        icu_procedure.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days-end",
                            "valueInteger": int(end_offset)
                        })
                
                entries.append({"resource": icu_procedure})
            
            # Add ICU flag to patient if they had any ICU stays
            if icu_count > 0:
                patient_resource.setdefault('extension', []).append({
                    "url": "http://mimic.mit.edu/fhir/StructureDefinition/ever-in-icu",
                    "valueBoolean": True
                })
                
                # Add ICU admission count
                patient_resource.setdefault('extension', []).append({
                    "url": "http://mimic.mit.edu/fhir/StructureDefinition/icu-stay-count",
                    "valueInteger": icu_count
                })
            
            # 4. Lab events as Observations
            lab_obs_count = 0
            patient_labs = df_labevents[df_labevents['subject_id'] == patient_id]
            
            # Limit to reasonable number (500 per patient)
            labs_to_process = patient_labs.head(500)
            for lab_idx, (_, lab_row) in enumerate(labs_to_process.iterrows(), 1):
                lab_obs_count += 1
                
                itemid = lab_row.get('itemid')
                if pd.isna(itemid):
                    continue
                
                obs_resource = {
                    "resourceType": "Observation",
                    "id": make_id("lab", patient_id, lab_idx),
                    "subject": {"reference": f"Patient/{patient_resource['id']}"},
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://mimic.mit.edu/itemid",
                            "code": str(itemid)
                        }]
                    }
                }
                
                # Add value
                value = lab_row.get('valuenum')
                if pd.notnull(value):
                    try:
                        obs_resource["valueQuantity"] = {"value": float(value)}
                        # Add unit if available
                        unit = lab_row.get('valueuom') or lab_row.get('unit')
                        if pd.notnull(unit):
                            obs_resource["valueQuantity"]["unit"] = str(unit)
                    except:
                        obs_resource["valueString"] = str(value)
                
                # Add timestamp and offset
                charttime = lab_row.get('charttime')
                if pd.notnull(charttime) and t0:
                    offset = compute_offset_days(t0, charttime)
                    if offset is not None:
                        obs_resource.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days",
                            "valueInteger": int(offset)
                        })
                
                entries.append({"resource": obs_resource})
            
            # 5. Chart events as Observations
            chart_obs_count = 0
            patient_charts = df_chartevents[df_chartevents['subject_id'] == patient_id]
            
            charts_to_process = patient_charts.head(500)
            for chart_idx, (_, chart_row) in enumerate(charts_to_process.iterrows(), 1):
                chart_obs_count += 1
                
                itemid = chart_row.get('itemid')
                if pd.isna(itemid):
                    continue
                
                obs_resource = {
                    "resourceType": "Observation",
                    "id": make_id("chart", patient_id, chart_idx),
                    "subject": {"reference": f"Patient/{patient_resource['id']}"},
                    "status": "final",
                    "code": {
                        "coding": [{
                            "system": "http://mimic.mit.edu/itemid",
                            "code": str(itemid)
                        }]
                    }
                }
                
                value = chart_row.get('valuenum')
                if pd.notnull(value):
                    try:
                        obs_resource["valueQuantity"] = {"value": float(value)}
                        unit = chart_row.get('valueuom') or chart_row.get('unit')
                        if pd.notnull(unit):
                            obs_resource["valueQuantity"]["unit"] = str(unit)
                    except:
                        obs_resource["valueString"] = str(value)
                
                charttime = chart_row.get('charttime')
                if pd.notnull(charttime) and t0:
                    offset = compute_offset_days(t0, charttime)
                    if offset is not None:
                        obs_resource.setdefault('extension', []).append({
                            "url": "http://your-research.org/fhir/StructureDefinition/event-offset-days",
                            "valueInteger": int(offset)
                        })
                
                entries.append({"resource": obs_resource})
            
            # 6. Diagnoses as Conditions
            patient_diagnoses = df_diagnoses[df_diagnoses['subject_id'] == patient_id]
            for diag_idx, (_, diag_row) in enumerate(patient_diagnoses.iterrows(), 1):
                icd_code = diag_row.get('icd_code')
                if pd.isna(icd_code):
                    continue
                    
                cond_resource = {
                    "resourceType": "Condition",
                    "id": make_id("cond", patient_id, diag_idx),
                    "subject": {"reference": f"Patient/{patient_resource['id']}"},
                    "code": {
                        "coding": [{
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": str(icd_code)
                        }]
                    },
                    "clinicalStatus": {
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active"
                        }]
                    }
                }
                entries.append({"resource": cond_resource})
            
            # 7. Add mortality outcome flag
            mortality_flag = mortality_data.get(patient_id, 0)
            if mortality_flag == 1:
                patient_resource.setdefault('extension', []).append({
                    "url": "http://mimic.mit.edu/fhir/StructureDefinition/hospital-mortality-outcome",
                    "valueBoolean": True
                })
            
            # 8. Add ICU admission flag if not already added
            icu_flag = icu_data.get(patient_id, 0)
            if icu_flag == 1 and icu_count == 0:
                # This should not happen if detailed_icu_data is correct
                patient_resource.setdefault('extension', []).append({
                    "url": "http://mimic.mit.edu/fhir/StructureDefinition/icu-admission-outcome",
                    "valueBoolean": True
                })
            
            # 9. Create Bundle
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
            
            # Save to file
            output_file = os.path.join(OUT_DIR, f"patient_{patient_id}.json")
            with open(output_file, 'w') as f:
                json.dump(bundle, f, indent=2)
            
            total_converted += 1
            patient_stats.append({
                'patient_id': patient_id,
                'admissions': admission_count,
                'icu_stays': icu_count,
                'lab_observations': lab_obs_count,
                'chart_observations': chart_obs_count,
                'total_observations': lab_obs_count + chart_obs_count,
                'diagnoses': len(patient_diagnoses),
                'mortality': mortality_flag,
                'icu_admission': icu_flag
            })
            
            if total_converted % 200 == 0:
                logging.info(f"  Converted {total_converted}/{len(selected_patient_ids)} patients...")
                
        except Exception as e:
            logging.error(f"Error converting patient {patient_id}: {e}")
            continue
    
    # Step 8: Create manifest and summary
    logging.info("\nStep 7: Creating manifest and summary...")
    
    # Shuffle patients for client assignment
    random.Random(RANDOM_SEED).shuffle(selected_patient_ids)
    clients = {}
    for i in range(N_CLIENTS):
        client_id = f"client_{i+1}"
        client_patients = selected_patient_ids[i::N_CLIENTS]
        clients[client_id] = client_patients
    
    # Create manifest
    manifest = {
        "conversion_info": {
            "target_patients": TARGET_PATIENTS,
            "actual_patients": total_converted,
            "conversion_date": datetime.now().isoformat(),
            "source": "MIMIC-IV v2.2",
            "selection_criteria": "Patients with admissions ≥1 day and observations"
        },
        "client_distribution": {k: len(v) for k, v in clients.items()},
        "data_statistics": {
            "min_observations_per_patient": MIN_OBSERVATIONS,
            "min_admissions_per_patient": MIN_ADMISSIONS
        }
    }
    
    with open(os.path.join(OUT_DIR, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create patient-to-client mapping
    patient_to_client = {}
    for client_id, patient_list in clients.items():
        for pid in patient_list:
            patient_to_client[str(pid)] = client_id
    
    with open(os.path.join(OUT_DIR, "patient_to_client_map.json"), 'w') as f:
        json.dump(patient_to_client, f, indent=2)
    
    # Create statistics summary with mortality and ICU data
    if patient_stats:
        stats_df = pd.DataFrame(patient_stats)
        
        # Calculate mortality and ICU rates
        mortality_count = int(stats_df['mortality'].sum())
        mortality_rate = mortality_count / len(stats_df) * 100 if len(stats_df) > 0 else 0
        
        icu_count = int(stats_df['icu_admission'].sum())
        icu_rate = icu_count / len(stats_df) * 100 if len(stats_df) > 0 else 0
        
        # Calculate average ICU stays for those with ICU admissions
        icu_patients = stats_df[stats_df['icu_admission'] == 1]
        total_icu_stays = int(stats_df['icu_stays'].sum())
        avg_icu_stays = icu_patients['icu_stays'].mean() if not icu_patients.empty else 0
        
        summary = {
            "total_patients_converted": total_converted,
            "observation_statistics": {
                "average_per_patient": float(stats_df['total_observations'].mean()),
                "median_per_patient": float(stats_df['total_observations'].median()),
                "minimum": int(stats_df['total_observations'].min()),
                "maximum": int(stats_df['total_observations'].max()),
                "std_dev": float(stats_df['total_observations'].std())
            },
            "admission_statistics": {
                "average_per_patient": float(stats_df['admissions'].mean()),
                "patients_with_multiple_admissions": int((stats_df['admissions'] > 1).sum())
            },
            "icu_statistics": {
                "icu_patients_count": icu_count,
                "icu_admission_rate": float(icu_rate),
                "total_icu_stays": total_icu_stays,
                "average_icu_stays_per_icu_patient": float(avg_icu_stays),
                "patients_with_multiple_icu_stays": int((stats_df['icu_stays'] > 1).sum())
            },
            "diagnosis_statistics": {
                "average_per_patient": float(stats_df['diagnoses'].mean()),
                "patients_with_diagnoses": int((stats_df['diagnoses'] > 0).sum())
            },
            "outcome_statistics": {
                "mortality_count": mortality_count,
                "mortality_rate": float(mortality_rate),
                "mortality_and_icu": int((stats_df['mortality'] & stats_df['icu_admission']).sum()),
                "mortality_rate_in_icu_patients": float(
                    (stats_df[stats_df['icu_admission'] == 1]['mortality'].sum() / 
                     max(1, len(icu_patients))) * 100
                )
            },
            "data_quality": {
                "patients_with_10plus_obs": int((stats_df['total_observations'] >= 10).sum()),
                "patients_with_50plus_obs": int((stats_df['total_observations'] >= 50).sum()),
                "patients_with_100plus_obs": int((stats_df['total_observations'] >= 100).sum())
            },
            "conversion_metrics": {
                "total_time_minutes": round((datetime.now() - start_time).total_seconds() / 60, 1),
                "patients_per_minute": round(total_converted / max(1, (datetime.now() - start_time).total_seconds() / 60), 1)
            }
        }
        
        with open(os.path.join(OUT_DIR, "conversion_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logging.info("\n" + "="*60)
        logging.info("✅ CONVERSION COMPLETE!")
        logging.info("="*60)
        logging.info(f"Patients converted: {total_converted}")
        logging.info(f"Average observations per patient: {summary['observation_statistics']['average_per_patient']:.1f}")
        logging.info(f"Mortality rate: {summary['outcome_statistics']['mortality_rate']:.2f}% ({mortality_count} deaths)")
        logging.info(f"ICU admission rate: {summary['icu_statistics']['icu_admission_rate']:.2f}% ({icu_count} ICU patients)")
        logging.info(f"Total ICU stays: {summary['icu_statistics']['total_icu_stays']}")
        logging.info(f"Average ICU stays per ICU patient: {summary['icu_statistics']['average_icu_stays_per_icu_patient']:.1f}")
        logging.info(f"Patients with ≥10 observations: {summary['data_quality']['patients_with_10plus_obs']}")
        logging.info(f"Patients with ≥50 observations: {summary['data_quality']['patients_with_50plus_obs']}")
        logging.info(f"Total time: {summary['conversion_metrics']['total_time_minutes']} minutes")
        logging.info(f"Speed: {summary['conversion_metrics']['patients_per_minute']} patients/minute")
        logging.info(f"Output directory: {OUT_DIR}")
    
    logging.info(f"\n🎉 Your {TARGET_PATIENTS} patient FHIR dataset with proper ICU data is ready for interoperability analysis!")

if __name__ == "__main__":
    main()
