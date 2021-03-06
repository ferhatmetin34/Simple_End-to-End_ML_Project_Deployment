import preprocessing_functions as pf 
import config
import warnings

warnings.simplefilter(action='ignore')

data = pf.load_data(config.PATH_TO_DATASET)

data = pf.drop_column(data,config.COLUMNS_TO_DROP)


X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)

count_encoder_= pf.count_encoder(config.COUNT_ENCODE,config.OUTPUT_COUNT_ENCODER,X_train)
X_train_enc=pf.encode_cats(config.OUTPUT_COUNT_ENCODER,X_train)
X_test_enc=pf.encode_cats(config.OUTPUT_COUNT_ENCODER,X_test)


mean_encoder_=pf.mean_encoder(config.MEAN_ENCODE,config.OUTPUT_MEAN_ENCODER,X_train_enc,y_train)
X_train_enc=pf.encode_cats(config.OUTPUT_MEAN_ENCODER,X_train_enc)
X_test_enc=pf.encode_cats(config.OUTPUT_MEAN_ENCODER,X_test_enc)

onehot_encoder_=pf.ohe_encoder(config.ONEHOT_ENCODE,config.OUTPUT_ONEHOT_ENCODER,X_train_enc)
X_train_enc=pf.encode_cats(config.OUTPUT_ONEHOT_ENCODER,X_train_enc)
X_test_enc=pf.encode_cats(config.OUTPUT_ONEHOT_ENCODER,X_test_enc)

scaler=pf.train_scaler(X_train_enc,config.OUTPUT_SCALER_PATH)
X_train_scaled=pf.scale_features(X_train_enc,config.OUTPUT_SCALER_PATH)
X_test_scaled=pf.scale_features(X_test_enc,config.OUTPUT_SCALER_PATH)

pf.train_model(X_train_scaled,y_train,config.OUTPUT_MODEL_PATH )
print("finished training")
