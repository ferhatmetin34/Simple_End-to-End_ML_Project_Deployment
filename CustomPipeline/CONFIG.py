PATH_TO_DATASET = "data_cleaned.csv"
COLUMNS_TO_DROP= ['brut_m2', 'floor', 'const_type'] 
COUNT_ENCODE= ["location",'room', 'age', 'heating']
ONEHOT_ENCODE= ["furniture","in_site"]
TARGET="price"
FEATURES=['net_m2', 'room', 'age', 'no_of_floor', 'heating', 'due', 'deposit',
       'no_of_bathroom', 'no_of_wc', "furniture_Boş","in_site_Hayır", 'location']
