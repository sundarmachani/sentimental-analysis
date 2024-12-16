import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download required NLTK data
nltk.download('vader_lexicon')


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.predictions = []
        self.actual_labels = []

    def predict_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        if scores['compound'] > 0.05:
            return 'Positive'
        elif scores['compound'] < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_text(self, text, actual_sentiment=None):
        predicted_sentiment = self.predict_sentiment(text)

        if actual_sentiment:
            self.predictions.append(predicted_sentiment)
            self.actual_labels.append(actual_sentiment)

        return predicted_sentiment

    def get_metrics(self):
        if not self.predictions or not self.actual_labels:
            return "No predictions made yet!"

        return {
            'accuracy': accuracy_score(self.actual_labels, self.predictions),
            'precision': precision_score(self.actual_labels, self.predictions, average='weighted', zero_division=0),
            'recall': recall_score(self.actual_labels, self.predictions, average='weighted', zero_division=0),
            'f1': f1_score(self.actual_labels, self.predictions, average='weighted', zero_division=0)
        }


# Interactive testing
def main():
    analyzer = SentimentAnalyzer()

    while True:
        print("\n=== Sentiment Analysis System ===")
        print("1. Analyze a single text")
        print("2. Analyze text with actual sentiment")
        print("3. View metrics")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            text = input("\nEnter text to analyze: ")
            sentiment = analyzer.analyze_text(text)
            print(f"\nPredicted Sentiment: {sentiment}")

        elif choice == '2':
            text = input("\nEnter text to analyze: ")
            print("\nEnter actual sentiment:")
            print("1. Positive")
            print("2. Negative")
            print("3. Neutral")

            sentiment_map = {'1': 'Positive', '2': 'Negative', '3': 'Neutral'}
            actual = input("Choose (1-3): ")

            if actual in sentiment_map:
                predicted = analyzer.analyze_text(text, sentiment_map[actual])
                print(f"\nActual Sentiment: {sentiment_map[actual]}")
                print(f"Predicted Sentiment: {predicted}")
            else:
                print("Invalid choice!")

        elif choice == '3':
            metrics = analyzer.get_metrics()
            if isinstance(metrics, dict):
                print("\nModel Performance Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-Score: {metrics['f1']:.4f}")
            else:
                print(metrics)

        elif choice == '4':
            print("\nThank you for using the Sentiment Analyzer!")
            break

        else:
            print("\nInvalid choice! Please try again.")


if __name__ == "__main__":
    main()
