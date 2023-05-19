import pandas as pd
import logging
import argparse
from preprocessing_helpers import classic_nlp_preprocessing, replace_html_number_codes


# Logging level configuration
logging.basicConfig(level=logging.DEBUG, format='%(process)d - %(levelname)s - %(message)s')

# General global variables
drop_list = ['url', 'author', 'publisher', 'header_image', 'raw_description', 'scraped_at']
text_cols = ['title', 'short_description', 'description']


def process_data(input_path: str = "nlp_data/cnbc_news_dataset.csv", drop_list: list = drop_list, text_cols: list = text_cols, remove_stop_words: bool = True, do_lemmatization: bool = True) -> None:
    """
    Apply preprocessing after replacing html number codes.
    Save to the same directory as the input path.
    """

    df = pd.read_csv(input_path, header=0, low_memory=False)
    df.drop(drop_list, axis=1, inplace=True)

    # Replace HTML number codes
    for col in text_cols:
        df[col] = df[col].apply(replace_html_number_codes)

    df = classic_nlp_preprocessing(df, text_cols, remove_stop_words=remove_stop_words, do_lemmatization=do_lemmatization, newsSpace=input_path.split('\\')[-1] == 'newsSpace.csv')
    df.to_csv(input_path.replace(".csv", "_processed.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply pre-processing for NLP training/inference')
    parser.add_argument('-i', '--input_path', type=str, default='nlp_data/cnbc_news_dataset.csv', help='Relative path to input file')
    parser.add_argument('-d', '--drop_list', nargs='*', default=drop_list, help='List of columns to drop')
    parser.add_argument('-t', '--text_cols', nargs='*', default=text_cols, help='List of columns containing text that should be included for all preprocessing operations')
    parser.add_argument('-rs', '--remove_stopwords', type=bool, default=True, help='Whether stopwords should be removed')
    parser.add_argument('-dl', '--do_lemmatization', type=bool, default=True, help='Whether lemmatization should be done')

    args = parser.parse_args()
    process_data(args.input_path, args.drop_list, args.text_cols, args.remove_stopwords, args.do_lemmatization)
