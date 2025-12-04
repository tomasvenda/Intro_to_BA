# Business Analytics Project - Final Submission

This repository contains the pipeline for predicting bike sharing demand in NYC using clustering and an ensemble of machine learning models.

## File Descriptions

*   **`clustering_parquet_file.ipynb`**: Preprocesses the raw data and assigns cluster IDs. This is the foundation for the subsequent models.
*   **`time_series_extract.ipynb`**: Trains and generates predictions using **SARIMAX** (for departures) and **Prophet** (for arrivals).
*   **`rf_extraction.ipynb`**: Generates predictions using the **Random Forest** model.
*   **`mlp_extract.ipynb`**: Generates predictions using the **MLP (Neural Network)** model.
*   **`regression_extraction.ipynb`**: Generates predictions using a **Regularized Linear Regression** model.
*   **`voting.ipynb`**: The final ensemble script. It aggregates predictions from all the models above to produce the final result and answer our Research Question.

## Execution Order

To ensure all dependencies and prediction files are available, please run the notebooks in the following order:

1.  **Data Preparation**
    *   Run `clustering_parquet_file.ipynb` first. - This requires the file 'Data/Trips_2018_initial_data.parquet' which was too large to be included. This step can be skipped for now as its output is already present in the Data folder

2.  **Model Extraction** (Can be run in any order, but must be completed before Voting)
    *   `time_series_extract.ipynb`
    *   `rf_extraction.ipynb`
    *   `mlp_extract.ipynb`
    *   `regression_extraction.ipynb`

3.  **Final Submission**
    *   Run `voting.ipynb` **last**.
