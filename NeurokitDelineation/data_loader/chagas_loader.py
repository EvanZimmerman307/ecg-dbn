import os
import wfdb
import numpy as np

# Update as needed
CHANNELS_SEQ = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_LENGTH = 2934

def list_record_names(dir_path):
    records = []
    for file in os.listdir(dir_path):
        if file.endswith(".hea"):
            record_name = file[:-4]
            if os.path.isfile(os.path.join(dir_path, record_name + ".dat")):
                records.append(record_name)
    return records

def load_record(file_path):
    try:
        return wfdb.rdrecord(file_path)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None

def extract_signal(record):
    if record is not None and hasattr(record, 'p_signal'):
        return record.p_signal
    return None

def load_dbn_data(dir_path, max_records=None):
    record_names = list_record_names(dir_path)
    if max_records:
        record_names = record_names[:max_records]

    signals_list = []
    samp_freq = None
    num_leads = None
    dropped = 0

    for rec_name in record_names:
        full_path = os.path.join(dir_path, rec_name)
        record = load_record(full_path)
        ecg = extract_signal(record)
        if ecg is not None:
            if ecg.shape[0] < TARGET_LENGTH:
                dropped += 1
                continue
            if samp_freq is None:
                samp_freq = record.fs
            if num_leads is None:
                num_leads = ecg.shape[1]
            signals_list.append(ecg.T[:, :TARGET_LENGTH]) # transposing makes it num_leads x obs, we only take the first 2934 obs
        else:
            print(f"Skipping invalid record {rec_name}")

    print(f"Loaded {len(signals_list)} records. Dropped {dropped} that were too short.")

    signals_array = np.stack(signals_list)  # shape: (N, n_leads, TARGET_LENGTH)
    return signals_array, samp_freq, CHANNELS_SEQ



"""
I can either pad or truncate to uniform length.
Out of 3268 samples, 78 or so have a length less than 2934.
One has a min length of 31.
If I pad, a lot of samples get padded, this is not good as it will hurt the integrity of the sample.
3268 - 1084 samples get padded.

I can ignore all samples with a length less than 2934, and truncate the rest.
I will only ignore 78 samples this way, and truncating the longer samples to 2934 does
not really hurt the value of those samples.
"""