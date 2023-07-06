import argparse
import pandas as pd
from util.timeseries_engineering_helpers import (
    normalize_datetime,
    get_file_names,
    replace_with_monday,
)
import logging


# Logging level configuration
logging.basicConfig(
    level=logging.INFO, format="%(process)d - %(levelname)s - %(message)s"
)


def process_all_files(
    input_path: str = "modelling/output_data/",
    output_path="data_engineering/timeseries_data",
    freq: str = "W",
) -> None:
    """
    Takes all files in the input_path and creates csv files with
    topic number as column name and date as the index.
    Finally export to specified output_path with the file name being
    ts_original_file_name.csv.

    Parameters
    ----------
    input_path
        Directory to scan
    output_path
        Directory to save transformed files to
    freq
        Frequency of the data in the export.

    Returns: None, it just saves the results to output_path.
    """

    # Discover all files in input path and filter for csv files only.
    csv_file_names = get_file_names(input_path)

    for file in csv_file_names:
        df = pd.read_csv(file, low_memory=False, index_col=0)
        ts_df = get_topic_ts(df, freq=freq)
        file_name = file.split("/")[-1]
        logging.info(f"Processed file {file_name}, now exporting...")
        ts_df.to_csv(f"{output_path}/ts_{file_name}", index=True)


def get_topic_ts(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Return a DataFrame containing the columns as topics with the dates with dates as index.
    Datetime objects in the index are normalized to be at midnight.

    Parameters
    ----------
    df
        DataFrame with raw data to create a time series per topic from
    freq
        Frequency of the returned data.

    Returns: DataFrame containing the topics as columns, each with a time series in specified frequency.
    """

    # Make sure that weekly frequency always starts at mondays
    assert (
        len(freq) == 1 if freq[0] == "W" else True
    ), "This function does not support weekly frequency with start days other than monday!"

    df["published_at"] = pd.to_datetime(df["published_at"])
    df["published_at"] = df["published_at"].apply(normalize_datetime)

    # Get weeks monday for each date if frequence is weekly
    df["published_at"] = (
        replace_with_monday(df["published_at"]) if freq == "W" else df["published_at"]
    )

    min_date = df["published_at"].min()
    max_date = df["published_at"].max()

    df.set_index("published_at", inplace=True)
    freq = "W-MON" if freq == "W" else freq
    date_range = pd.date_range(min_date, max_date, freq=freq)

    accumulated_scores = pd.DataFrame(index=date_range)
    logging.debug(accumulated_scores.index)

    for index, row in df.iterrows():
        if pd.notna(row["topic_class"]):
            topic_class = eval(row["topic_class"])
        else:
            continue

        # Accumulate the scores for each topic number
        for topic, score in topic_class:
            if topic in accumulated_scores.columns:
                if index in accumulated_scores.index:
                    accumulated_scores.loc[index, topic] = (
                        accumulated_scores.loc[index, topic] + score
                    )
                else:
                    accumulated_scores.loc[index, topic] = score
            else:
                accumulated_scores = pd.concat(
                    [accumulated_scores, pd.DataFrame(columns=[topic])], axis=1
                )
                accumulated_scores[topic] = 0
                accumulated_scores.loc[index, topic] = score

    accumulated_scores.sort_index(axis=0, inplace=True)
    accumulated_scores.sort_index(axis=1, inplace=True)

    return accumulated_scores.fillna(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process files to generate time series data."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="modelling/output_data/",
        help="Relative path to input files",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="data_engineering/timeseries_data",
        help="Relative path for output time series data",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="D",
        help="Frequency for time series data (e.g., D for daily)",
    )

    args = parser.parse_args()
    process_all_files(args.input_path, args.output_path, args.freq)
