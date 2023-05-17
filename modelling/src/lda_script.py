import pandas as pd
import gensim
import argparse

def run_lda_experiment(num_topics:int, num_passes:int, data_input_path:str, 
                       data_output_path:str, dataset_name:str,input_column:str) -> None:
    """This function icludes the workflow to perform Latent Dirichlet Allocation. 
    It loads the data from the specified path, transforms it, trains an LDA model and performs inference.
    Results are saved in the specified output path.

    Keyword arguments:
    num-topics -- the number of topics for lda
    num-passes -- the number of passes/epochs for training
    data-input-path -- relative path to the input data (like ./.../filename.csv)
    data-output-path -- relative path to the output data (like ./.../)
    dataset-name -- name of the dataset used for training
    input-column -- name of the column with the text data
    """
    # Load processed data from csv
    df = pd.read_csv(data_input_path)
    # Convert into list of lists
    processed_docs = []
    for i in list(df[input_column]):
        if type(i)==str:
            processed_docs.append(eval(i))
    # Create a dictionary from 'processed_docs' containing the number of times a word appears in the training
    dictionary = gensim.corpora.Dictionary(processed_docs)  
    # Bag-of-words model for each document (dictionary per doc reporting how many words and how many times those words appear)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Start Model Training
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = num_topics, 
                                   id2word = dictionary,                                    
                                   passes = num_passes,
                                   workers = 8)
    
    # Save Training results 
    model_name = f'lda_model_topics_{num_topics}_passes_{num_passes}_on_{input_column}'
    model_path = f'./models/LDA/{model_name}'
    lda_model.save(model_path)
    # Run Inference 
    def make_inference_on_doc(doc):
        # Data preprocessing step for the unseen document
        if type(doc)==str:
            bow_vector = dictionary.doc2bow(eval(doc))
            return lda_model[bow_vector]
    # Add topic labels to original dataset and save to output path
    df['topic_class'] = df[input_column].apply(lambda x: make_inference_on_doc(x))
    df.to_csv(f"{data_output_path}data_{model_name}_{dataset_name}")


# Retrive passed arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-topics", type=int)
    parser.add_argument("--num-passes", type=int)
    parser.add_argument("--data-input-path", type=str)
    parser.add_argument("--data-output-path", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--input-column", type=str)
    args = parser.parse_args()
    # Start experiment
    run_lda_experiment(
        args.num_topics, 
        args.num_passes,
        args.data_input_path,
        args.data_output_path,
        args.dataset_name,
        args.input_column
    )