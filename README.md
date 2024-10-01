# Sentiment Polarity Classification

## Description
This project involves creating a binary sentiment polarity classifier using positive and negative movie reviews. The dataset used contains 5,331 positive and 5,331 negative reviews. The classifier is built using both Neural Networks and Random Forest approaches, and the performance of each model is evaluated using key metrics such as precision, recall, F1-score, and confusion matrix.

## Dataset
The dataset used for this project is the Cornell Movie Review Data:
- [Download Link](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
- [ReadMe File](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.README.1.0.txt)

The data is divided as follows:
- Training Set: First 4,000 positive and 4,000 negative reviews
- Validation Set: Next 500 positive and 500 negative reviews
- Test Set: Final 831 positive and 831 negative reviews

## Models
### 1. Neural Network:
- 3-layer feed-forward neural network with ReLU activations and dropout for regularization.
- Sigmoid activation used in the output layer for binary classification.

### 2. Random Forest Classifier:
- RandomForestClassifier with 100 estimators for binary classification.

## Evaluation Metrics
The performance of the models is evaluated using the following metrics:
- **Confusion Matrix**
- **Precision**
- **Recall**
- **F1-score**

## Requirements
To run this code, you need the following software and packages installed:

### Software
- Python 3.8+

### Python Libraries
- `pandas` (Data manipulation and analysis)
- `scikit-learn` (Machine learning algorithms and evaluation metrics)
- `tensorflow` (Neural network model building)
- `matplotlib` (Plotting results)
- `seaborn` (Visualizing confusion matrix heatmaps)
- `gdown` (Downloading files from Google Drive, if applicable)

Ensure that the required Python libraries are installed using the following command:
   ```bash
   pip install pandas scikit-learn tensorflow matplotlib seaborn
   ```

## Instructions to Run

1. Download the dataset using the provided [link](https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz).
2. Unzip the dataset and place the `rt-polarity.pos` and `rt-polarity.neg` files in your project directory.

3. Open the provided Jupyter Notebook `NLP_Sentiment_Analysis.ipynb` in Google Colab or your local Jupyter environment.

4. Run each cell in sequence to execute the sentiment polarity classification code.

5. The cells include the following main steps:
   - **Data Preprocessing**: Loading positive and negative movie reviews and splitting into training, validation, and test sets.
   - **TF-IDF Vectorization**: Transforming the text data into numerical feature vectors.
   - **Model Training**: Building and training models using Neural Networks and Random Forest classifiers.
   - **Model Evaluation**: Calculating performance metrics (precision, recall, F1-score) and plotting confusion matrix heatmaps.

6. After running the cells, you will be able to view the final performance metrics and confusion matrix plots directly in the notebook.

### Example Command to Open the Notebook Locally:
```bash
jupyter notebook NLP_Sentiment_Analysis.ipynb
```

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
