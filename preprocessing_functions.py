import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
import joblib
from feature_engine.encoding import MeanEncoder,CountFrequencyEncoder,DecisionTreeEncoder,OneHotEncoder,OrdinalEncoder

def load_data(df_path):
    return pd.read_csv(df_path)

def drop_column(df,columns):
    df=df.drop(columns=columns,axis=1)
    return df 

def divide_train_test(df, target):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target,axis=1),
                                                        df[target],
                                                        test_size=0.3,
                                                        random_state=0)
    return X_train, X_test, y_train, y_test

def count_encoder(var,output_path, df):
    encoder=CountFrequencyEncoder(variables=var)
    encoder.fit(df)
    joblib.dump(encoder, output_path)
    return encoder 

def mean_encoder(var,output_path, X,y):
    encoder=MeanEncoder(variables=var)
    encoder.fit(X,y)
    joblib.dump(encoder, output_path)
    return encoder

def ohe_encoder(var,output_path, df):
    encoder=OneHotEncoder(variables=var)
    encoder.fit(df)
    joblib.dump(encoder, output_path)
    return  encoder
    
def encode_cats(encoder,df):
    encoder=joblib.load(encoder)
    return encoder.transform(df)


def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler

def scale_features(df, scaler):
    scaler = joblib.load(scaler) 
    return scaler.transform(df)

def train_model(df, target, output_path):
    
    model =LGBMRegressor(max_depth=3,reg_alpha=0.5,reg_lambda=50)  
    model.fit(df, target)
    joblib.dump(model, output_path)
    
    return None

def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)

