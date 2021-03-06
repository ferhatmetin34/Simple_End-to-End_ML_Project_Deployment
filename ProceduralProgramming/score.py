import preprocessing_functions as pf
import config

def predict(data):
    
   
    data=pf.encode_cats(config.OUTPUT_COUNT_ENCODER,data)

    data=pf.encode_cats(config.OUTPUT_MEAN_ENCODER,data)

    data=pf.encode_cats(config.OUTPUT_ONEHOT_ENCODER,data)

    data=pf.scale_features(data,config.OUTPUT_SCALER_PATH)

    predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)
        
    return predictions

if __name__ == '__main__':
    
    from math import sqrt
    import numpy as np
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    import warnings
    warnings.simplefilter(action='ignore')

    data = pf.load_data(config.PATH_TO_DATASET)

    data = pf.drop_column(data,config.COLUMNS_TO_DROP)

    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    pred = predict(X_test)

    print('test mse: {}'.format(int(mean_squared_error(y_test,pred))))
    print('test rmse: {}'.format(int(sqrt(mean_squared_error(y_test, pred)))))
    print('test r2: {}'.format(r2_score(y_test,pred)))
    print()
        
    