# Sentiment Analysis

## Overview

This project implements a real-time sentiment analysis system using Natural Language Processing (NLP) techniques. It analyzes user-input text and predicts the sentiment as Positive, Negative, or Neutral. The system also allows users to provide actual sentiments, enabling comparison between predicted and actual results.

## Features

- Real-time sentiment analysis of user input text
- Comparison of predicted sentiment with user-provided actual sentiment
- Performance metrics calculation (Accuracy, Precision, Recall, F1-score)
- Interactive command-line interface for easy usage

## Requirements

- Python 3.6+
- NLTK
- scikit-learn
- pandas

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sundarmachani/sentimental-analysis.git
   cd sentiment-analysis-project
   ```

2. Install the required packages:
   ```
   pip install nltk scikit-learn pandas
   ```

3. Download the necessary NLTK data:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

Run the script:
```
python sentiment-analyzer.py
```

Follow the on-screen menu to:
1. Analyze a single text
2. Analyze text and provide actual sentiment
3. View performance metrics
4. Exit the program

## How It Works

The sentiment analysis is performed using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer. The system calculates a compound score for each input text and classifies it as follows:
- Positive: compound score > 0.05
- Negative: compound score < -0.05
- Neutral: -0.05 ≤ compound score ≤ 0.05

## Performance Metrics

The system calculates the following metrics based on user-provided actual sentiments:
- Accuracy: Overall correctness of predictions
- Precision: Proportion of correct positive predictions
- Recall: Proportion of actual positives correctly identified
- F1-score: Harmonic mean of precision and recall

## Future Improvements

- Implement more advanced NLP models (e.g., BERT, RoBERTa)
- Add support for sentiment analysis of larger text documents
- Develop a graphical user interface (GUI)
- Integrate with social media APIs for real-time sentiment analysis of posts/tweets

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.