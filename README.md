# Data Exploration Project
University: DHBW Mannheim
Course: WWI21DSB

## Goal of this project
The aim of the project is to answer the question of whether an abstract representation of world events can be used to improve forecasting of financial markets. To this end, we initially wanted to create a latent representation of our world using information extracted from news articles, which was then supposed to serve as part of the input for a state of the art forecasting model like DeepAR. Later, data from the Global Database of Events, Language and Tone (GDELT) was used for the event information, as depicted in the diagram below.

![project diagram](https://github.com/jhb300/project_submission_haiperformer/blob/main/documentation/project_architecture_v2_hai_performer.png?raw=true)

## Members of the group
- Jan Henrik Bertrand (8556462)
- David Hoffmann (2571020)
- Marc Grün (9603221)
- Felix Noll (9467152)

## How to use this repository

### Requirements
This project was developed using Python 3.9. Install the requirements using the following command:
```bash
pip install -r requirements.txt
```

### Project Structure
The project is divided into 3 separate workstreams: 

1. Data Collection
2. Data Engineering
3. Modelling

The contents and scope of each is described in the respective subsections below:

### Data Collection:
- cameo_translation: Translation of the [CAMEO](https://en.wikipedia.org/wiki/Conflict_and_Mediation_Event_Observations) in the GDELT dataset, to natural language.
- financial_ts: Auxiliary time series containing macro-economic information.
- web_crawl_links: Links to GDELT data that needs to be crawled (2014 & 2015).
- downloaded_files: GDELT daily reports, that are scraped in the gdelt_web_crawl.ipynb notebook.
- gdelt_web_crawl.ipynb: Notebook to execute the retrieval of the GDELT data between 2014 and 2015.

### Data Engineering
- exploration: All exploratory notebooks that do preprocessing and transformation on the auxiliary/related cnbc_news dataset as well as on GDELT. The track_record notebook contains the track record measures the market performance of the model. The track_record_comparison notebook compares two models that were trained with the same hyperparameters, but one with the GDelt dataset and the other one without it.
- financial_ts: Central directory containing all processed financial data (indices and related time series), serving as a single point of truth w.r.t. the financial data for the modelling workstream.
- nlp_data: Contains the preprocessed cnbc_news dataset, ready for clustering by the modelling workstream.
- src: Contains preprocessing scripts for the cnbc_news dataset and for transforming the clustered cnbc_news data into time series with weekly frequency. The util directory contains helper scripts that are used by both the exploratory notebooks and the preprocessing scripts.
- time_series: Contains time series indicating the intensity of topic clusters in the cnbc_news dataset.
- __init__.py: This directory is a Python package, to make it possible to import across the subfolders (e.g. importing scripts from util in exploratory notebooks).

### Documentation
- project_architecture_v1.png: Contains a diagram of the initial version of the project idea.
- project_architecture_v2.png: Contains a diagram of the adapted version of the project including GDELT.
- project_report_hai_performer.pdf: The detailed documentation of the project. Including business use case, motivation and explanation of all workstreams.
- track_record.csv: Comparison of the models performance with the performance of the index that the model trades on.
- track_record_gdelt.csv: Comparison of the models performance with the performance of the index that the model trades on using a once trained model with GDelt as related time series.
- track_record_no_gdelt.csv: Comparison of the models performance with the performance of the index that the model trades on using a once trained model without GDelt as related time series.

### Modelling
- backtests: Includes backtests of the final model with 6 to 10 backtest windows using a single model as well as a backtest with individual models trained for each backtest window.
- config: Holds the database files used to keep track of experiments and their results as well as the past_rts_cols.json file used to specify the related time series for training.
- exploration: Notebooks for the initial setup and experiments of the nlp modelling (topic extraction) focusing on LDA and time series forecasting focusing on DeepAR.
- models: Includes LDA model artifacts.
- output_data: Output of the nlp modelling, containing the news dataset with an additional attribute that holds a vector representation of the topics for each article.
- src: Python scripts for model training and for plotting forecasts.
- modelling_environment.ipynb: Primary notebook for running forecast experiments and model training with DeepAR, MQ-CNN, and DeepState.
