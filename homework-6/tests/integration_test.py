import os
import pandas as pd
from datetime import datetime
from batch import prepare_data, get_storage_options

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)
  
def test_save_to_s3():
    data = [
            (None, None, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2), dt(1, 10)),
            (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
            (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
        ]

    categorical = ['PUlocationID', 'DOlocationID']
    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)
        
    year = 2021
    month = 1
        
    df.to_parquet(
        f"s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/input.parquet",
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=get_storage_options()
    )
    
    
def test_batch_integration():
    year = 2021
    month = 1
    os.putenv("INPUT_FILE_PATTERN", "s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/input.parquet")
    os.putenv("OUTPUT_FILE_PATTERN", "s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet")
    os.putenv("S3_ENDPOINT_URL", "http://localhost:4566")
    os.system(f"python batch.py {year} {month}")
    df_output = pd.read_parquet(f"s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet", storage_options=get_storage_options())
    
    print(df_output['predicted_duration'].sum())
    
    expected = pd.DataFrame([
        ('2021/01_0', 23.052085),
        ('2021/01_1',46.236612)
    ], columns = ['ride_id', 'predicted_duration'])
    
    
    pd.testing.assert_frame_equal(df_output, expected)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    