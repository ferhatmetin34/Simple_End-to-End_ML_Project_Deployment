PATH_TO_DATASET = "data_cleaned.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'lightgbm_regressor.pkl'
OUTPUT_COUNT_ENCODER= "count_encoder.pkl"
OUTPUT_MEAN_ENCODER="mean_encoder.pkl"
OUTPUT_ONEHOT_ENCODER="onehot_encoder.pkl" 
COLUMNS_TO_DROP= ['brut_m2', 'floor', 'const_type'] 
COUNT_ENCODE= ["location"]
MEAN_ENCODE= ['room', 'age', 'heating']
ONEHOT_ENCODE= ["furniture","in_site"]
TARGET="price"
FEATURES=['net_m2', 'room', 'age', 'no_of_floor', 'heating', 'due', 'deposit',
       'no_of_bathroom', 'no_of_wc', 'in_site', 'furniture', 'location']
