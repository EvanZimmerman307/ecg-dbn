from delineation_feature_experimentation import DBN_delineation_dataset, dbn_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Standard Initialization
    data_path = 'neurokit_delineation/output/DBN_ecg_all_features.csv'
    wfdb_path = '/Users/evanzimm/GitHub/python-example-2025/dbn_dataset'
    dbn_data = DBN_delineation_dataset(data_path, wfdb_path)
    dbn_data.cols_to_numpy()

    # Do any feature engineering you need to on dbn_data.self.delineation_data

    # st_segment, Time between offset of S-wave to onset of T-wave,Float64,msec 
    # could be used to calculate offset of S since we know onset of T. CALCULATED.    
    dbn_data.delineation_data['s_offsets'] = dbn_data.delineation_data['t_onsets'] - dbn_data.delineation_data['st_segment']

    # qt_interval, Time between onset of Q-wave to offset of T-wave,Float64,msec 
    # I could do t_offsets +- qt_interval and get onset of q-wave. CALCULATED.  
    dbn_data.delineation_data['q_onsets'] = dbn_data.delineation_data['t_offsets'] - dbn_data.delineation_data['qt_interval']

    # List the features you want in your dataset
    features = ['p_onsets', 'p_offsets', 'r_offsets', 't_onsets', 't_offsets', 's_offsets', 'q_onsets']
    anchor = "r_onsets"

    # create the beatwise dataframe
    dbn_data.create_beatwise_data(features, anchor, 50) # 50 gave an error, 100 was best

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

    dbn.fit_dbn(df_train)

    acutal, predicted = dbn.inference(df_test)

    y_true = df_test[('chagas', 0)]

    y_pred = [p.values.argmax() for p in predicted]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Precision: {prec}, Recall: {rec}, F1 Score: {f1}")

