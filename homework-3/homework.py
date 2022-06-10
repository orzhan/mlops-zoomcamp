import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
import datetime
import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    logger = get_run_logger()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger = get_run_logger()
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"The MSE of validation is: {mse}")
    return

def get_paths(date):
    train_date = (date.replace(day=1) - datetime.timedelta(days=1)).strftime("%Y-%m")
    val_date = (date.replace(day=1) - datetime.timedelta(days=2)).strftime("%Y-%m")
    train_path = f'./data/fhv_tripdata_{train_date}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date}.parquet'
    return train_path, val_path

@flow
def main(date=None):
    if date is None:
        date = datetime.date.today()
    else:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    train_path, val_path = get_paths(date)
    
    logger = get_run_logger()
    logger.info(f"Using files: {train_path}, {val_path}")

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    with open(f'./artifacts/model-{date.strftime("%Y-%m-%d")}.pkl', 'wb') as fout:
        pickle.dump(lr, fout)
    
    with open(f'./artifacts/dv-{date.strftime("%Y-%m-%d")}.pkl', 'wb') as fout:
        pickle.dump(dv, fout)
    
    run_model(df_val_processed, categorical, dv, lr)

#main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="model_training",
    flow=main,
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Europe/Moscow"),
)