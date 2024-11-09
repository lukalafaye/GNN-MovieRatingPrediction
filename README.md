# Movie Rating Prediction

This project predicts movie ratings for 610 users using a Graph Neural Network (GNN). The project also includes a traditional matrix factorization approach with hyperparameter tuning and a movie metadata scraper. It is designed to showcase and compare different methods for movie rating prediction.

## Project Structure

- `presentation.pdf`: The presentation for the project, explaining the methodology and results.
- `report.pdf`: The detailed report with analysis, results, and conclusions.
- `README.md`: This file, providing an overview of the project and instructions for running it.
- `src/`: The source code directory containing the following files:
  - `drive-data/`: Contains data related to movies and pre-trained models.
    - `movie-metadata.csv`: A CSV file with metadata on 5,000 movies.
    - `tensor.pth`: Pretrained model for generating movie embeddings.
  - `final_preds.npy`: The predicted movie ratings for 610 users, generated using the GNN model.
  - `gnn-cuda-run-on-collab.ipynb`: A Jupyter notebook to run the Graph Neural Network model on Google Colab (requires specific CUDA version).
  - `matrix_factorization_hyperparameters.ipynb`: A Jupyter notebook to implement and tune the matrix factorization approach for movie rating prediction.
  - `requirements.txt`: A file to install the necessary dependencies before running the notebooks.
  - `scrape-movie-metadata.ipynb`: A Jupyter notebook that uses the OMDB API to scrape movie metadata.

## Installation & Setup

1. **Install dependencies**:  
  Ensure you have Python installed, then install the required packages by running:

  ```bash
  pip install -r requirements.txt
  ``` 

## Running the notebooks:
- To run the GNN model, open the `gnn-cuda-run-on-collab.ipynb` notebook in Google Colab.
- To explore matrix factorization, open the `matrix_factorization_hyperparameters.ipynb` notebook.
- To scrape additional movie metadata, use the `scrape-movie-metadata.ipynb` notebook.

## Using the pretrained model:
The file `tensor.pth` contains pre-trained movie embeddings. These embeddings can be used in your own models or analysis.

## Predictions:
The `final_preds.npy` contains the predicted movie ratings for 610 users. You can load this file using `numpy` to evaluate the GNN predictions.

## Notes
Made with love by Luka Lafaye de Micheaux and Newman Chen from PSL.
