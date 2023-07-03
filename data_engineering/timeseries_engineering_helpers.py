from datetime import datetime
import pandas as pd
import os


def normalize_datetime(dt: datetime) -> datetime:
    """
    Normalize datetime object to be in UTC time and at midnight.

    Parameters
    ----------
    dt
        Datetime object to normalize
        
    Returns: Datetime object with desired properties.
    """

    if isinstance(dt, datetime):
        dt = dt.tz_convert('UTC')
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt


def get_file_names(directory: str) -> list:
    """
    Return names of all csv files in a given directory.
    
    Parameters
    ----------
    directory
        Directory to scan
        
    Returns: List of all csv file names.
    """

    cwd = os.getcwd()
    full_directory_path = os.path.join(cwd, directory)
    file_names = []
    for root, dirs, files in os.walk(full_directory_path):
        file_names.extend(iter(files))
    
    # Concatenate the input_path with the file names
    file_names = list(map(lambda file_name: os.path.join(directory, file_name), file_names))

    return list(filter(lambda x: x[-4:] == ".csv", file_names))


def replace_with_monday(datetime_column: pd.Series) -> pd.Series:
    """
    Get the Monday of the same week for each datetime.

    Parameters
    ----------
    datetime_column
        Series containing datetime objects on dates that should be replaced with
        the Monday of the same week.
        
    Returns: Series with Mondays only.
    """

    return datetime_column - pd.to_timedelta(datetime_column.dt.dayofweek, unit='D')
