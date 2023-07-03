import pandas as pd
import xml.etree.ElementTree as Xet
import argparse
import logging


# Logging level configuration
logging.basicConfig(level=logging.DEBUG, format='%(process)d - %(levelname)s - %(message)s')


others = [f"other{count}" for count in range(1)]
columns = [
    'source',
    'url',
    'title',
    'image',
    'category',
    'description',
    'rank',
    'pubdate',
    'video',
]

def convert(input: str="nlp_data/newsSpace.xml", cols: list=columns) -> None:
    """
    Convert the file of the input path from XML to CSV format and save it to 
    the original directory.

    Parameters
    ----------
    input
        Input file to convert
    cols
        Columns to extract from the XML file
        
    Returns: None, it just saves the results to the input directory.
    """

    # Parsing the XML file
    rows = []
    xmlparse = Xet.parse(input)
    root = xmlparse.getroot()
    root_list = list(root.iter())[1:]

    logging.debug(f"Starting to assemble the data for the DataFrame. Starting with: {root_list[0]}, Text: {root_list[0].text}")

    count = 0
    while len(root_list) > 8:
        if count % 1000 == 0 and count != 0:
            percentage_done = (count / len(root_list)) * 100
            logging.info(f"Processed another thousand elments, now at {round(percentage_done, 6)}%")

        row = []
        for col in cols:
            if root_list[0].tag == col:
                row.append(root_list.pop(0).text)
            else:
                row.append(None)
        rows.append(row)
        count += 1

    df = pd.DataFrame(data=rows, columns=cols)
    logging.debug(f"Finished assembly of DataFrame, now starting to export: \n{df.head()}")
    df.to_csv(input.replace(".xml", ".csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a file from XML format to CSV format.')
    parser.add_argument('-i', '--input_path', type=str, default='nlp_data/newsSpace.xml', help='Relative path to input file.')
    parser.add_argument('-c', '--columns', type=list, default=columns, help='Columns that the csv should contain in the end.')

    args = parser.parse_args()
    convert(args.input_path, args.columns)