  
import pandas as pd
import numpy as np
from preprocessors import Pipeline
import CONFIG

pipeline=Pipeline(target=CONFIG.TARGET,
                    columns_to_drop=CONFIG.COLUMNS_TO_DROP,
                    count_encode=CONFIG.COUNT_ENCODE,
                    onehot_encode=CONFIG.ONEHOT_ENCODE,
                    features=CONFIG.FEATURES)

if __name__ == '__main__':

    data=pd.read_csv(CONFIG.PATH_TO_DATASET)

    data=pipeline.drop_column(data)
    pipeline.fit(data)

    print("model performance")

    pipeline.evaluate_model()

    print("Predictions")

    predictions=pipeline.predict(data)
    print(predictions)

    

    