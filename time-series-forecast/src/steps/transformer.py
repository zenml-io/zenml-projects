from zenml.steps import step, Output
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

@step
def transformer(data: pd.DataFrame) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray #df = pd.DataFrame
):
    df = data.copy()
    cardinal_directions = {'N': 0.0,
                            'NNE': 0.0068591439603377154,
                            'NE': 0.013700834628155488,
                            'ENE': 0.0205599785884932,
                            'E': 0.027419122548830915,
                            'ESE': 1.9634954084936207,
                            'SE': 2.356194490192345,
                            'SSE': 2.748893571891069,
                            'S': 3.141592653589793,
                            'SSW': 3.5342917352885173,
                            'SW': 3.9269908169872414,
                            'WSW': 4.319689898685966,
                            'W': 4.71238898038469,
                            'WNW': 5.105088062083414,
                            'NW': 5.497787143782138,
                            'NNW': 5.8904862254808625}

    for direction in cardinal_directions:
            df.loc[df["Direction"] == direction, "Direction"] = cardinal_directions[direction]
    
    df['Direction'] = df['Direction'].astype(float)
    df["v1"] =  df['Speed'] * np.cos(np.array(df["Direction"]))
    df["v2"] =  df['Speed'] * np.sin(np.array(df["Direction"]))
    
    df = df.drop(['Direction', 'Speed'], axis=1)

    X = np.array(df[['v1','v2']])
    y = np.array(df['Total'])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    return X_train, X_test, y_train, y_test 
