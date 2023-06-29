import pandas as pd
from gluonts.model.forecast import SampleForecast


def get_df_from_Sample_Forecast(forecast: SampleForecast) -> pd.DataFrame:
    """
    Get a Pandas DataFrame from a GluonTS SampleForecast object.

    Parameters
    ----------
    forecast
        A SampleForecast obj for one year-
        
    Returns: Pandas DataFrame
    """

    samples = forecast.samples
    ns, h = samples.shape
    start_date_period = forecast.start_date
    start_date_timestamp = start_date_period.to_timestamp()
    dates = pd.date_range(start_date_timestamp, freq='W-MON', periods=h)
    return pd.DataFrame(samples.T, index=dates)


def shift_weekly_series_to_monday(series: pd.Series) -> pd.Series:
    """
    Convert a time series in weekly frequency into weekly frequency starting monday.

    Parameters
    ----------
    series
        Pandas Series to convert to weekly frequency starting monday.
        
    Returns: Pandas Series
    """

    series.index = pd.to_datetime(series.index)
    if series.index[0].dayofweek == 0:
        return series
    
    # Shift the series index to start on Monday
    shifted_series = series.shift(1, freq='D')
    shifted_series = shifted_series.resample('W-MON').last()
    
    return shifted_series

