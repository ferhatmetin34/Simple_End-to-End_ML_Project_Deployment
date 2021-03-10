import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

from feature_engine.encoding import MeanEncoder,CountFrequencyEncoder,OneHotEncoder

class Pipeline:
    def __init__(self,target,columns_to_drop, count_encode,onehot_encode,features,test_size=0.3,random_state=0):

        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None

        self.features=features
        self.columns_to_drop=columns_to_drop
        self.count_encode=count_encode
        
        self.onehot_encode=onehot_encode

        self.target=target 
        self.test_size=test_size
        self.random_state=random_state

        self.scaler=MinMaxScaler()
        self.model=LGBMRegressor(max_depth=3,reg_alpha=0.5,reg_lambda=50)

      
        self.count_encoder=CountFrequencyEncoder(variables=count_encode)
        self.onehot_encoder=OneHotEncoder(variables=onehot_encode)

    def drop_column(self,data):
        data=data.drop(columns=self.columns_to_drop,axis=1)
        return data

    def fit(self,data):
        #data=data.drop(columns=self.columns_to_drop,axis=1)

        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(data,
                    data[self.target],
                    test_size=self.test_size,
                    random_state=self.random_state)

        self.count_encoder.fit(self.X_train)
    
        self.onehot_encoder.fit(self.X_train)

        self.X_train=self.count_encoder.transform(self.X_train)
        self.X_train=self.onehot_encoder.transform(self.X_train)

        self.X_test=self.count_encoder.transform(self.X_test)
        self.X_test=self.onehot_encoder.transform(self.X_test)

        self.scaler.fit(self.X_train[self.features])
        self.X_train=self.scaler.transform(self.X_train[self.features])
        self.X_test=self.scaler.transform(self.X_test[self.features])

        print(self.X_train.shape, self.X_test.shape)

        self.model.fit(self.X_train,self.y_train)

        return self

    def transform(self,data):
        data=data.copy()
        #data=data.drop(columns=self.columns_to_drop,axis=1)
        data=self.count_encoder.transform(data)
        data=self.onehot_encoder.transform(data)
        data=self.scaler.transform(data[self.features])

        return data

    def predict(self,data):
        #data=data.drop(columns=self.columns_to_drop,axis=1)
        
        data=self.transform(data)
        
        predictions=self.model.predict(data)

        return predictions

    def evaluate_model(self):
        pred=self.model.predict(self.X_train)
        print("train r2: {} ".format(r2_score(self.y_train,pred)))

        pred=self.model.predict(self.X_test)
        print("test r2: {} ".format(r2_score(self.y_test,pred)))

