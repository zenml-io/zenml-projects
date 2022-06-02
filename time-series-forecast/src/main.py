from google.oauth2 import service_account
from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseStepConfig
from zenml.repository import Repository
import pandas_gbq
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

step_operator = Repository().active_stack.step_operator

class BigQueryImporterConfig(BaseStepConfig):
    query: str = 'SELECT * FROM `computas_dataset.wind_forecast`'
    project_id: str = 'computas-project-345810'


@step
def bigquery_importer(config: BigQueryImporterConfig) -> pd.DataFrame:
    credentials = service_account.Credentials.from_service_account_file('./credentials.json')
    return pandas_gbq.read_gbq(config.query, project_id = config.project_id, credentials = credentials)

@step
def preparator(data: pd.DataFrame) -> Output(
    df = pd.DataFrame #np.ndarray #, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    df = data.drop(['Source_time','Lead_hours','ANM','Non_ANM', 'int64_field_0'],axis=1)
    df = df[df['Direction'].notna()]
    df = df[df['Total'].notna()]
    df['Speed'] =  df['Speed'].fillna(df['Speed'].median())

    return df

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
    #return df

#@step
#def splitter(data: pd.DataFrame) -> Output(
#    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
#):
#    X = np.array(data[['v1','v2']])
#    y = np.array(data['Total'])

#    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
#    return X_train, X_test, y_train, y_test 

@step(custom_step_operator=step_operator.name)
def trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    print("I am traning")
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RandomForestRegressor,
) -> float:
    
    y_pred = model.predict(X_test)
    score = r2_score(y_test,y_pred)
    print(f'R2 score: {score}')

    return score

@pipeline
def iris_pipeline(
    bigquery_importer,
    preparator,
    transformer,
#    splitter,
    trainer,
    evaluator,
):
    """Links all the steps together in a pipeline"""
    data = bigquery_importer() 
    prepared_data = preparator(data=data) #, X_test, y_train, y_test = preparator(data=data)
    #transformed_data = transformer(data=prepared_data)
    X_train, X_test, y_train, y_test = transformer(data=prepared_data) #splitter(data=transformed_data)
    model = trainer(X_train=X_train,y_train=y_train)
    evaluator(X_test=X_test, y_test=y_test, model=model)

pipeline = iris_pipeline(
    bigquery_importer=bigquery_importer(),
    preparator=preparator(),
    transformer=transformer(),
    #splitter=splitter(),
    trainer=trainer(),
    evaluator=evaluator(),
)

pipeline.run()
