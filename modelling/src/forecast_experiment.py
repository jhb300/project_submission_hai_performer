# Importing the necessary packages
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.evaluation import Evaluator, backtest_metrics
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation.metrics import mape
from gluonts.model.estimator import Estimator

import pandas as pd
import numpy as np
from datetime import datetime

def run_experiment(
    input_paths: list,
    target_col: str,
    prediction_length: int,
    windows: int,
    past_rts_col: list,
    estimator: Estimator,
) -> dict:
    """
    Loads data and trains an estimator model with the passes parameters 
    and returns the aggregated metrics and timeseries. 

    Parameters
    ----------
    input_paths 
        List of data input paths (date column must be Unnamed: 0).
    target_col 
        Name of the target column.
    prediction_length
        Length of the prediction horizon.
    windows
        Number of forecast test windows. 
    past_rts_col 
        List of the past rts column (dynamic) names.
    estimator
        Pass an Estimator object.
        
        
    Returns
    -------
    forecasts, 
        Yields the forecast timeseries.
    tss
         Yields the corresponding ground truth series.
    agg_metrics
        Dictionary with agregated metrics.
    """
    ###############
    #  Load Data  #
    ###############
    
    df_input_list = []
    for file_path in input_paths:
        # Load input data
        temp_df = pd.read_csv(file_path)
        # Format DataFrame
        # temp_col_map = {i:f"tts_{i}" for i in temp_df.columns if i.isnumeric()}
        temp_df = temp_df.rename(columns={'Unnamed: 0': 'Week'})
        temp_df['Week'] = temp_df['Week'].apply(
            lambda x: datetime.fromisoformat(x))
        temp_df = temp_df.set_index('Week')
        # Rename Columns
        temp_col_map = {i: f"ts_{i}" for i in temp_df.columns}
        temp_df = temp_df.rename(columns=temp_col_map)
        df_input_list.append(temp_df)
    # Join to single DataFrame
    ts_df = df_input_list[0]
    for remaining_df in df_input_list[1:]:
        ts_df = ts_df.join(remaining_df)
    ts_df = ts_df.reset_index()
    # Convert to GluonTS Dataset
    dataset = PandasDataset(
        ts_df,
        target=target_col,
        timestamp='Week',
        freq='W',
        past_feat_dynamic_real=past_rts_col
    )
    # Split the data for training and testing
    training_data, test_gen = split(dataset, offset=-(prediction_length*windows))
    test_data = test_gen.generate_instances(
        prediction_length=prediction_length, windows=windows)
    
    ############################
    #  Training and Inference  #
    ############################
    
    predictor = estimator.train(training_data)

    # Make inference
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data.dataset,
        predictor=predictor,
        num_samples=100,
    )
    forecasts_single = list(forecast_it)
    forecasts = list(predictor.predict(test_data.input))
    tss = list(ts_it)
    
    # Compute metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts_single)

    return forecasts, tss, agg_metrics
