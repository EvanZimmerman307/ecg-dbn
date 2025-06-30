import pandas as pd
import numpy as np
from chagas_delineation_loader import load_dbn_data
from pgmpy.models import DynamicBayesianNetwork as DBN
from sklearn.model_selection import train_test_split
import logging
from pgmpy.inference import DBNInference
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score


"""
The idea behind this script is to write a general framework for learning and testing DBNs
using data containing different wave delineation features produced from neurokit wave delineation.

After defining the features from the wave delineation results. The rest should be automated by this script.
"""

def parse_array_str(s):
    """
    Parse a string representing an array of numbers into a NumPy array.
    The string is expected to be of the form "[ 153 490 824 ...]" or "[153, 490, 824, ...]".
    """
    # Remove the surrounding brackets and any leading/trailing whitespace
    s = s.strip().strip('[]')
    
    # Replace commas with a space, if present
    s = s.replace(',', ' ')
    
    # Split the string by whitespace to extract the numbers
    number_strings = [num for num in s.split() if num]
    
    # Convert the list of strings to a numpy array of floats (or ints if appropriate)
    try:
        arr = np.array([float(num) for num in number_strings])
        # Optionally convert to int if all values are integers
        if np.all(arr.astype(int) == arr):
            arr = arr.astype(int)
    except Exception as e:
        print("Error parsing string to array:", s, e)
        arr = np.array([])
    
    return arr

class DBN_delineation_dataset():

    def __init__(self, delineation_output_path, wfdb_data_path):
        # delineation results in a dataframe
        # NOTE once I create delineation_data I think I can replace delineation_results
        self.delineation_data = pd.read_csv(delineation_output_path) # maybe storing 2 big dataframes is not good???
        self.wfdb_dataset_path = wfdb_data_path
        self.static_vars = []
        self.dynamic_vars = []
        self.max_t = None
        self.training_df = None
        return

    def cols_to_numpy(self):
        """
        when the csv is read into a df the onset/offset lists are strings and they need to be numpy arrays
        """
        self.delineation_data['p_onsets'] = self.delineation_data['p_onsets'].apply(parse_array_str)
        self.delineation_data['p_offsets'] = self.delineation_data['p_offsets'].apply(parse_array_str)
        self.delineation_data['r_peaks'] = self.delineation_data['r_peaks'].apply(parse_array_str)
        self.delineation_data['r_onsets'] = self.delineation_data['r_onsets'].apply(parse_array_str)
        self.delineation_data['r_offsets'] = self.delineation_data['r_offsets'].apply(parse_array_str)
        self.delineation_data['t_onsets'] = self.delineation_data['t_onsets'].apply(parse_array_str)
        self.delineation_data['t_offsets'] = self.delineation_data['t_offsets'].apply(parse_array_str)
        return
    
    def get_beatwise_features(self, features, anchor_data, window_ms=150):
        """
        Returns a list of dictionaries, each representing one beat.
        Each dict has binary indicators for whether an event is within the defined window
        around the r_onset of that beat.

        r_onsets: list of r_onsets for a row
        features: a dictionary where {key: feature_name, val: row[feature_name]}
        """
        
        """
        The use of & requires that for each individual element, 
        both conditions must be true; but np.any() then only requires at least one of those elements to pass the test.
        """
        def in_window(event_array, center, window_ms):
            if event_array is None or len(event_array) == 0:
                return 0
            return int(np.any((event_array >= center - window_ms) & (event_array <= center + window_ms)))
        
        beat_features = []
        
        for r in anchor_data: # for each point in the anchor's data for one ecg reading, check if other onsets/offsets occur around that point
            feat = {feature_name: in_window(features[feature_name], r, window_ms) for feature_name in features.keys()}
            beat_features.append(feat)
        
        return beat_features
    
    def create_beatwise_data(self, features, anchor, window):
        """
        features: a list of feature names in the delineation df that you want to extract beatwise data for
        anchor: the feature to anchor timesteps around (string)
        """
        all_beats_features = []
        self.dynamic_vars = features

        for idx, row in self.delineation_data.iterrows():
            feature_row_map = {feature: row[feature] for feature in features} # {key: feature_name, val: row[feature_name]}
            anchor_data = row[anchor] 

            # Generate beat-wise features
            beat_feats = self.get_beatwise_features(feature_row_map, anchor_data, window_ms=window)

            # Add the  'sample_idx' to keep track
            # for example we skip 192, sample_idx != 192, so row_idx != 192
            for b in beat_feats:
                b['row_idx'] = row['sample_idx'] 
            
            all_beats_features.extend(beat_feats)
        
        # Convert to a DataFrame of beat-wise features
        self.delineation_data = pd.DataFrame(all_beats_features)
        
        print("Created the beatwise dataframe!")
        print(self.delineation_data.head())
        return
    
    def load_chagas_age_gender(self):
        """
        Load the non-ecg variables from the original dataset. Note that doing delineation drops some records, so the labels from
        load_dbn_data don't match 1-1 with the records in the delineation output (can't just append columns). 
        We can use the sample indices to index the labels that correspond to records that exist.
        """
        ecg_array, samp_freq, channel_seq, label_list, gender_list, age_list = load_dbn_data(self.wfdb_dataset_path)

        label_list = np.array(label_list)
        gender_list = np.array(gender_list)
        age_list = np.array(age_list)

        # add labels by indexing 'row_idx' to make sure that labels line up with the right row_idx
        self.delineation_data['chagas'] = label_list[self.delineation_data['row_idx']]
        self.delineation_data['gender'] = gender_list[self.delineation_data['row_idx']]
        self.delineation_data['age'] = age_list[self.delineation_data['row_idx']]

        # make age discrete
        self.delineation_data['age'] = pd.cut(self.delineation_data['age'], bins=4, labels=False, right=False)

        self.static_vars = ['age', 'gender']

        print("Added chagas, age, and gender to the beatwise dataframe!")
        print(self.delineation_data.head())
        return
    
    def load_chagas(self):
        ecg_array, samp_freq, channel_seq, label_list, gender_list, age_list = load_dbn_data(self.wfdb_dataset_path)

        label_list = np.array(label_list)
        self.delineation_data['chagas'] = label_list[self.delineation_data['row_idx']]

        print("Added chagas to the beatwise dataframe!")
        print(self.delineation_data.head())
        return
    
    def create_empty_training_data(self):
        max_steps = self.delineation_data.groupby('row_idx').agg("count")
        self.max_t = max(max_steps['p_onsets']) # should be 21

        columns = []
        for t in range(self.max_t): # t = 0, 1, ..., 21
            for var in self.dynamic_vars + self.static_vars:
                columns.append((var, t))

        df_all_obs = pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns))

        self.training_df = df_all_obs

        print("Created Empty Training Dataframe!")
        return
    
    def populateTrainingDF(self):
        latent_var = 'chagas'
        all_row_idxs = self.delineation_data['row_idx'].unique() # these are the samples indices in the original

        # this loops through each ECG Reading
        for row in all_row_idxs:
            obs_df = self.delineation_data[self.delineation_data['row_idx'] == row].reset_index(drop=True) # get the beatwise data specific to a reading
            obs_length = len(obs_df)
            obs_data = {}

            for t in range(self.max_t):
                if t < obs_length:
                    # Explicitly assign observed ECG wave delineation data
                    for var in self.dynamic_vars:
                        obs_data[(var, t)] = obs_df.loc[t, var]
                else:
                    # Explicitly pad ECG signals with NaNs
                    for var in self.dynamic_vars:
                        obs_data[(var, t)] = np.nan

                # Explicitly repeat latent variable and static vars at ALL timesteps
                obs_data[(latent_var, t)] = obs_df.loc[0, latent_var]
                for var in self.static_vars:
                    obs_data[(var, t)] = obs_df.loc[0, var]
                
            # Append observation explicitly as a single row
            df_obs = pd.DataFrame([obs_data])
            self.training_df = pd.concat([self.training_df, df_obs], ignore_index=True)
        
        print("Populated Training dataframe (flattened the beatwise data)")
        return
    
    def col_type_to_int64(self, col):
        # Use pd.to_numeric to force conversion; errors='coerce' will turn non-numeric entries into NaN,
        # and then .astype('Int64') will convert the column to the pandas nullable integer type.
        return pd.to_numeric(col, errors='coerce').astype("Int64")
    

class dbn_model():

    def __init__(self, static_vars, ecg_vars):
        self.dbn = DBN()
        # Static variables influence latent state at each timestep separately (no temporal edges for static vars)
        self.static_vars = static_vars

        # Edges from static variables to HiddenState at time 0
        for var in self.static_vars:
            self.dbn.add_edge((var, 0), ('chagas', 0))

        # Edges from static variables to HiddenState at time 1
        for var in self.static_vars:
            self.dbn.add_edge((var, 1), ('chagas', 1))

        # Latent state temporal edges (usual)
        self.dbn.add_edge(('chagas', 0), ('chagas', 1))

        self.ecg_vars = ecg_vars

        for wave in self.ecg_vars:
            self.dbn.add_edges_from([
                (('chagas', 0), (wave, 0)),
                (('chagas', 1), (wave, 1))
            ])
    
    def fit_dbn(self, data):
        self.dbn.fit(data)
        
        print("Learned DBN params!")
        return
    
    def inference(self, data):
        # Set up logging to a file
        logging.basicConfig(
            filename='inference.log',
            filemode='w',  # Overwrite file on each run; use 'a' to append
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO  # or DEBUG for more details
        )
        logger = logging.getLogger(__name__)

        dbn_infer = DBNInference(self.dbn)  # your trained DBN

        predictions = []
        true_labels = []

        for idx, row in data.iterrows():
            logger.info(f"Processing row index: {idx}")
            
            # Build evidence dict: all observed columns except the latent variable
            evidence = {}
            valid_timesteps = []

            # Identify valid timesteps based on non-NaN ECG data
            for col in data.columns:
                var_name, t_step = col  # deconstruct column tuple
                if var_name not in ['chagas', 'gender', 'age'] and pd.notna(row[col]):
                    valid_timesteps.append(t_step)
            
            if not valid_timesteps:
                logger.info(f"Row {idx}: No valid timesteps found, skipping row.")
                continue  # or handle the case with no evidence
            
            max_valid_timestep = max(valid_timesteps)
            logger.info(f"Row {idx}: Max valid timestep = {max_valid_timestep}")

            # Build the evidence dict for valid timesteps
            for col in data.columns:
                var_name, t_step = col
                if t_step <= max_valid_timestep and var_name != 'chagas':
                    evidence[(var_name, t_step)] = row[col]
            
            logger.info(f"Row {idx}: Raw evidence: {evidence}")

            # Clean up the evidence by converting all values to plain ints
            evidence_clean = {}
            for key, value in evidence.items():
                if pd.notna(value):
                    try:
                        evidence_clean[key] = int(value)
                    except Exception as e:
                        logger.error(f"Error converting evidence for {key}: {value}")
                        raise e
            logger.info(f"Row {idx}: Cleaned evidence: {evidence_clean}")

            # Instead of looping over every t_step, just query the last valid timestep:
            try:
                logger.info(f"Row {idx}: Querying for ('chagas', {max_valid_timestep})")
                query_result = dbn_infer.query(
                    variables=[('chagas', max_valid_timestep)],
                    evidence=evidence_clean,
                )
                logger.info(f"Row {idx}, timestep {max_valid_timestep}: Prediction: {query_result[('chagas', max_valid_timestep)]}")
            except Exception as e:
                logger.error(f"Error during query at row {idx}, timestep {max_valid_timestep}")
                raise e

            predictions.append(query_result[('chagas', max_valid_timestep)])
            
            # Store the true label; assuming the label is the same at all timesteps, use time 0
            true_label = row[('chagas', 0)]
            true_labels.append(true_label)
            logger.info(f"Row {idx}: True label: {true_label}")

        logger.info("Inference complete.")

        print("Finished Inference!")
        return true_labels, predictions


if __name__ == "__main__":
    """
    This script works!!!!!
    """

    # Standard Initialization
    data_path = 'NeurokitDelineation/output/DBN_ecg_all_features.csv'
    wfdb_path = 'dbn_dataset'
    dbn_data = DBN_delineation_dataset(data_path, wfdb_path)
    dbn_data.cols_to_numpy()

    # Do any feature engineering you need to on dbn_data.self.delineation_data

    # List the features you want in your dataset
    features = ['p_onsets', 'p_offsets', 'r_offsets', 't_onsets', 't_offsets']
    anchor = "r_onsets"

    # create the beatwise dataframe
    dbn_data.create_beatwise_data(features, anchor, 200)

    # add the non-ecg variables
    dbn_data.load_chagas_age_gender()

    dbn_data.create_empty_training_data()
    dbn_data.populateTrainingDF()

    # Train/Test split
    class_labels = dbn_data.training_df[('chagas', 0)]

    # Perform explicit 80/20 stratified train-test split
    df_train, df_test = train_test_split(
        dbn_data.training_df,
        test_size=0.2,
        stratify=class_labels,
        random_state=42  # reproducibility
    )

    for col in df_train.columns:
        df_train[col] = dbn_data.col_type_to_int64(df_train[col])
    
    for col in df_test.columns:
        df_test[col] = dbn_data.col_type_to_int64(df_test[col])
    
    dbn = dbn_model(dbn_data.static_vars, dbn_data.dynamic_vars)

    df_train = df_train.fillna(0) # How could something even be NAN?
    dbn.fit_dbn(df_train)

    acutal, predicted = dbn.inference(df_test)

    y_true = df_test[('chagas', 0)]

    y_pred = [p.values.argmax() for p in predicted]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Precision: {prec}, Recall: {rec}, F1 Score: {f1}")





    


    






    





