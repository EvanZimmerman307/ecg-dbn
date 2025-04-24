import json
import os
import pandas as pd
from data_loader.mediconnect_loader import load_mediconnect_data
from data_loader.ukbiobank_loader import load_ukbiobank_data
from data_loader.mimic_loader import load_mimic_data
from data_loader.chagas_loader import load_dbn_data
from modules.r_peak_detection import clean_ecg, detect_r_peaks
from modules.delineation import delineate_ecg
from modules.feature_extraction import ECGFeatureExtractor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

DATASET = 'DBN'

def process_ecg_data(features = None):
    # features is a list of features you want to write to csv

    # Load ECG data from their custom data loader
    # if DATASET == 'MEDICONNECT':
    #     ecg_array, samp_freq, channel_seq = load_mediconnect_data(config['DATASETS']['MEDICONNECT']['DICOM_DIR'])
    # elif DATASET == 'UKBIOBANK':
    #     ecg_array, samp_freq, channel_seq = load_ukbiobank_data(config['DATASETS']['UKBIOBANK']['DIR_PATH'])
    # elif DATASET == 'MIMIC':
    #     ecg_array, samp_freq, channel_seq = load_mimic_data(config['DATASETS']['MIMIC']['DIR_PATH'])
    # else:
    #     raise ValueError('Dataset not supported and no data loader is created')
    dbn_dataset_path = "/Users/evanzimm/GitHub/python-example-2025/dbn_dataset"
    ecg_array, samp_freq, channel_seq = load_dbn_data(dbn_dataset_path)

    # clean the ecgs to remove any noise and baseline wandering
    cleaned_ecg = clean_ecg(ecg_array, sampling_rate=samp_freq)

    # detect the R peaks for every lead of the ecgs separately
    r_peaks = detect_r_peaks(cleaned_ecg, sampling_rate=samp_freq, dataset=DATASET)

    # delineate the cleaned ecg into its components
    delineation_results = delineate_ecg(cleaned_ecg, r_peaks, sampling_rate=samp_freq)

    # extract ecg baseline features with related intervals
    feature_extractor = ECGFeatureExtractor(cleaned_ecg, r_peaks, delineation_results, samp_freq, channel_seq)
    features_df, dropped_idxs = feature_extractor.extract_features() # extract dropped_indxs

    if features is not None:
        # get the columns with the main baseline features
        # features_baseline = features_df[['sample_idx', 'lead', 'heart_rate', 'r_peaks', 'pr_interval', 'qrs_complex',
                                     #'qt_interval', 'rr_interval', 'st_segment']]
        features_baseline = features_df[features]
    else:
        features_baseline = features_df
    
    # extract the ecg annotations with metadata of features
    annotations_df = feature_extractor.generate_annotations(features_df)
    annotations_df = annotations_df.convert_dtypes()

    # create output directory if it does not already exist
    output_directory = config['OUTPUT_DIRECTORY']
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    data_file_name = "TEMP_FEATURES"

    # save the features and their metadata to a CSV file
    features_baseline.to_csv(os.path.join(output_directory, f'{DATASET}_{data_file_name}.csv'),
                             float_format='%.3f', index=False)
    annotations_df.to_csv(os.path.join(output_directory, 'annotations.csv'), index=False)

    # save the dropped indxs
    dropped_idxs_df = pd.DataFrame()
    dropped_idxs_df['dropped'] = dropped_idxs
    dropped_idxs_df.to_csv(os.path.join(output_directory, 'dropped.csv'), index=False)

if __name__ == '__main__':
    """
    was features = ['sample_idx', 'lead', 'heart_rate', 'r_peaks', 'pr_interval', 'qrs_complex',
                                     #'qt_interval', 'rr_interval', 'st_segment']
    """
    process_ecg_data()
