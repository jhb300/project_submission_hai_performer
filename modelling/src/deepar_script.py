# Importing the necessary packages
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.evaluation import Evaluator, backtest_metrics
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation.metrics import mape

import pandas as pd
import numpy as np
from datetime import datetime

prediction_length = 56
num_layers = 2
hidden_size = 40
max_epochs = 400


def run_deepar_experiment(
    input_paths: list,
    target_col: str,
    past_rts_col: list,
    trainer_kwargs: dict,
    context_length: int,
    prediction_length: int = 56,
    num_layers: int = 2,
    hidden_size: int = 40,
    weight_decay: float = 1e-8,
    lr: float = 1e-3,
) -> dict:
    """
    Loads data and trains a DeepAR (torch) model with the passes parameters 
    and returns the aggregated metrics. 

    Parameters
    ----------
    input_paths 
        List of data input paths (date column must be Unnamed: 0).
    target_col 
        Name of the target column.
    past_rts_col 
        List of the past rts column (dynamic) names.
    trainer_kwargs
        Arguments to pass to the torch Trainer.
    prediction_length
        Specifies the forecast horizon and backtest window len.
    num_layers 
        Number of hidden layers.
    hidden_size 
        Number of nodes per hidden layer.
    weight_decay
        Weight decay regularization parameter (default: 1e-8).
        
        
    Returns
    -------
    forecasts, 
        Yields the forecast timeseries.
    tss
         Yields the corresponding ground truth series.
    agg_metrics
        Dictionary with agregated metrics.
    """

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
    training_data, test_gen = split(dataset, offset=-prediction_length)
    test_data = test_gen.generate_instances(
        prediction_length=prediction_length, windows=1)

    # Train the model and make predictions
    predictor = DeepAREstimator(
        prediction_length=prediction_length,
        freq="W",
        num_layers=num_layers,  # Number of RNN layers (default: 2).
        hidden_size=hidden_size, # Number of RNN cells per layer (default: 40).
        weight_decay=weight_decay,
        context_length=context_length,
        lr = lr,
        trainer_kwargs=trainer_kwargs
    ).train(training_data)

    # Make inference
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data.dataset,
        predictor=predictor,
        num_samples=100,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # Compute metrics
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    return forecasts, tss, agg_metrics
