

# Sentiment Analysis Tool using Streamlit

This project implements a simple sentiment analysis tool using a pre-trained logistic regression model on the IMDb movie reviews dataset. The tool allows users to input a movie review, and it predicts whether the sentiment of the review is positive or negative.

## Requirements

- Python 3.6+
- Streamlit
- pandas
- numpy
- scikit-learn
- nltk

You can install the required Python libraries using the following command:
```
pip install streamlit pandas numpy scikit-learn nltk
```

Additionally, you need to download NLTK resources for natural language processing tasks. Run the following code in your Python environment before running the app:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Installation

1. Clone or download this repository to your local machine.
2. Install the required libraries as mentioned above.
3. Download the IMDb movie reviews dataset and save it as `Test.csv` in the project directory.

## Usage

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the Streamlit app using the following command:
   ```
   streamlit run app.py
   ```
4. The app will open in your default web browser. Enter a movie review in the text area and click the "Analyze" button to see the predicted sentiment (positive or negative) of the review.

## File Structure

- `app.py`: Main Python script containing the Streamlit app code.
- `Test.csv`: IMDb movie reviews dataset (not included in this repository, download and add it to the project directory).

## Acknowledgments

- The IMDb movie reviews dataset is used for sentiment analysis.
- NLTK and scikit-learn libraries are used for natural language processing and machine learning tasks.
