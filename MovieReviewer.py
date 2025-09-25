"""
Movie Review Sentiment Classifier using Naive Bayes Algorithm

This module implements a binary sentiment classifier that can determine whether
a movie review is positive or negative using the Naive Bayes algorithm.

The classifier is trained on the aclImdb dataset which contains:
- 25,000 training reviews (12,500 positive + 12,500 negative)
- 25,000 test reviews (12,500 positive + 12,500 negative)

Link to dataset: https://ai.stanford.edu/~amaas/data/sentiment/
"""

import os
import re
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import random

class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for binary sentiment analysis of movie reviews.
    
    This class implements the Naive Bayes algorithm from scratch, including:
    - Text preprocessing and tokenization
    - Vocabulary building from training data
    - Word probability calculation with Laplace smoothing
    - Sentiment prediction using Bayes' theorem
    - Model evaluation with precision, recall, and F1-score metrics
    """
    
    def __init__(self):
        """
        Initialize the Naive Bayes classifier with empty data structures.
        
        Attributes:
            vocabulary (set): Set of all unique words in the training data
            word_counts (dict): Count of each word for each class (positive/negative)
            class_counts (dict): Number of documents in each class
            total_words (dict): Total word count for each class
            class_priors (dict): Prior probability of each class
            vocab_size (int): Size of the vocabulary
        """
        # Initialize data structures for storing training information
        self.vocabulary = set()  # All unique words from training data
        self.word_counts = {'positive': defaultdict(int), 'negative': defaultdict(int)}  # Word frequency per class
        self.class_counts = {'positive': 0, 'negative': 0}  # Number of documents per class
        self.total_words = {'positive': 0, 'negative': 0}  # Total word count per class
        self.class_priors = {'positive': 0, 'negative': 0}  # Prior probability of each class
        self.vocab_size = 0  # Size of vocabulary for Laplace smoothing
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by converting to lowercase, removing special characters,
        and splitting into words.
        
        This method performs basic text cleaning to standardize the input text
        before feature extraction. It removes punctuation, numbers, and special
        characters while preserving only alphabetic characters and spaces.
        
        Args:
            text (str): Raw text input to be preprocessed
            
        Returns:
            List[str]: List of cleaned words from the input text
            
        Example:
            Input: "This movie is AMAZING!!! 5/5 stars."
            Output: ["this", "movie", "is", "amazing", "stars"]
        """
        # Convert to lowercase for case-insensitive processing
        text = text.lower()
        
        # Remove special characters, digits, and punctuation
        # Keep only letters (a-z, A-Z) and whitespace characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into individual words and remove empty strings
        # This handles multiple spaces and ensures clean word list
        words = [word for word in text.split() if word.strip()]
        
        return words
    
    def load_reviews(self, directory: str, label: str) -> List[str]:
        """
        Load all review files from a directory and return as list of texts.
        
        This method reads all .txt files from the specified directory and loads
        their contents into memory. It handles encoding issues gracefully and
        filters out empty files.
        
        Args:
            directory (str): Path to the directory containing review files
            label (str): Label for the reviews (e.g., 'positive', 'negative')
                        Used for logging purposes only
            
        Returns:
            List[str]: List of review texts loaded from the directory
            
        Note:
            - Only processes .txt files
            - Skips empty files
            - Uses UTF-8 encoding with error handling
        """
        reviews = []
        
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            # Only process text files
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                
                # Read file with UTF-8 encoding and ignore encoding errors
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read().strip()
                    
                    # Only add non-empty reviews to avoid processing empty files
                    if text:
                        reviews.append(text)
        
        # Log the number of reviews loaded for debugging
        print(f"Loaded {len(reviews)} {label} reviews from {directory}")
        return reviews
    
    def build_vocabulary(self, reviews: List[str]) -> None:
        """
        Build vocabulary from all reviews.
        
        This method processes all training reviews to create a comprehensive
        vocabulary of unique words. The vocabulary is used for feature extraction
        and probability calculations during training and prediction.
        
        Args:
            reviews (List[str]): List of all training review texts
            
        Note:
            - Vocabulary is built from preprocessed words only
            - Duplicate words are automatically handled by set operations
            - Vocabulary size is stored for Laplace smoothing calculations
        """
        # Process each review and extract unique words
        for review in reviews:
            # Preprocess the review text to get clean word list
            words = self.preprocess_text(review)
            # Add words to vocabulary set (automatically handles duplicates)
            self.vocabulary.update(words)
        
        # Store vocabulary size for probability calculations
        self.vocab_size = len(self.vocabulary)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def train(self, train_pos_dir: str, train_neg_dir: str) -> None:
        """
        Train the Naive Bayes classifier on positive and negative reviews.
        
        This method performs the complete training process:
        1. Loads training data from specified directories
        2. Builds vocabulary from all training reviews
        3. Counts word frequencies for each class
        4. Calculates class priors (P(positive) and P(negative))
        
        Args:
            train_pos_dir (str): Directory containing positive training reviews
            train_neg_dir (str): Directory containing negative training reviews
            
        Note:
            - Training data should be in .txt format
            - Each file should contain one review
            - Class priors are calculated as document counts / total documents
        """
        print("Loading training data...")
        
        # Load positive and negative reviews from their respective directories
        pos_reviews = self.load_reviews(train_pos_dir, 'positive')
        neg_reviews = self.load_reviews(train_neg_dir, 'negative')
        
        # Build vocabulary from all training reviews (both positive and negative)
        # This ensures we have a complete vocabulary for probability calculations
        all_reviews = pos_reviews + neg_reviews
        self.build_vocabulary(all_reviews)
        
        # Count words for each class - this is the core of Naive Bayes training
        print("Training classifier...")
        
        # Process positive reviews and count word frequencies
        for review in pos_reviews:
            words = self.preprocess_text(review)
            for word in words:
                # Only count words that are in our vocabulary
                if word in self.vocabulary:
                    self.word_counts['positive'][word] += 1
                    self.total_words['positive'] += 1
            # Count the number of positive documents
            self.class_counts['positive'] += 1
        
        # Process negative reviews and count word frequencies
        for review in neg_reviews:
            words = self.preprocess_text(review)
            for word in words:
                # Only count words that are in our vocabulary
                if word in self.vocabulary:
                    self.word_counts['negative'][word] += 1
                    self.total_words['negative'] += 1
            # Count the number of negative documents
            self.class_counts['negative'] += 1
        
        # Calculate class priors: P(positive) and P(negative)
        # These represent the probability of each class before considering any features
        total_docs = self.class_counts['positive'] + self.class_counts['negative']
        self.class_priors['positive'] = self.class_counts['positive'] / total_docs
        self.class_priors['negative'] = self.class_counts['negative'] / total_docs
        
        # Display training completion summary
        print(f"Training completed!")
        print(f"Positive reviews: {self.class_counts['positive']}")
        print(f"Negative reviews: {self.class_counts['negative']}")
        print(f"Class priors - Positive: {self.class_priors['positive']:.4f}, Negative: {self.class_priors['negative']:.4f}")
    
    def calculate_word_probability(self, word: str, class_label: str) -> float:
        """
        Calculate P(word|class) using Laplace smoothing.
        
        This method calculates the probability of observing a specific word
        given that the document belongs to a particular class. Laplace smoothing
        is applied to handle words that never appeared in the training data for
        a given class, preventing zero probabilities that would break the model.
        
        Formula: P(word|class) = (count(word, class) + 1) / (total_words_in_class + vocab_size)
        
        Args:
            word (str): The word to calculate probability for
            class_label (str): The class ('positive' or 'negative')
            
        Returns:
            float: The probability P(word|class) with Laplace smoothing applied
            
        Note:
            - Laplace smoothing prevents zero probabilities for unseen words
            - The smoothing factor is 1 (pseudo-count) added to numerator
            - Vocabulary size is added to denominator for normalization
        """
        # Get the count of this word in the specified class
        word_count = self.word_counts[class_label][word]
        # Get total word count in this class
        total_words_in_class = self.total_words[class_label]
        
        # Apply Laplace smoothing to prevent zero probabilities
        # Add 1 to numerator and vocabulary size to denominator
        probability = (word_count + 1) / (total_words_in_class + self.vocab_size)
        
        return probability
    
    def predict_single(self, review: str) -> Tuple[str, float, float]:
        """
        Predict sentiment for a single review.
        
        This method implements the core Naive Bayes prediction algorithm using
        Bayes' theorem. It calculates the probability of the review belonging
        to each class and returns the most likely class along with confidence scores.
        
        The prediction process:
        1. Preprocess the input review text
        2. Calculate log probabilities for each class using Bayes' theorem
        3. Convert back to regular probabilities with numerical stability
        4. Normalize probabilities to sum to 1
        5. Return the class with highest probability
        
        Args:
            review (str): The movie review text to classify
            
        Returns:
            Tuple[str, float, float]: (predicted_class, positive_prob, negative_prob)
                - predicted_class: 'positive' or 'negative'
                - positive_prob: Probability that review is positive (0-1)
                - negative_prob: Probability that review is negative (0-1)
                
        Note:
            - Uses log probabilities to prevent numerical underflow
            - Applies numerical stability techniques for overflow prevention
            - Only considers words present in the training vocabulary
        """
        # Preprocess the review text to get clean word list
        words = self.preprocess_text(review)
        
        # Initialize log probabilities with class priors
        # Using log space prevents numerical underflow when multiplying many small probabilities
        log_prob_positive = math.log(self.class_priors['positive'])
        log_prob_negative = math.log(self.class_priors['negative'])
        
        # Calculate log probabilities for each word using Bayes' theorem
        # P(class|review) ∝ P(class) × ∏ P(word|class)
        for word in words:
            # Only process words that exist in our vocabulary
            if word in self.vocabulary:
                # Calculate P(word|positive) and P(word|negative)
                word_prob_positive = self.calculate_word_probability(word, 'positive')
                word_prob_negative = self.calculate_word_probability(word, 'negative')
                
                # Add log probabilities (equivalent to multiplying regular probabilities)
                log_prob_positive += math.log(word_prob_positive)
                log_prob_negative += math.log(word_prob_negative)
        
        # Convert back to regular probabilities with numerical stability
        # Subtract the maximum log probability to prevent overflow when using exp()
        max_log_prob = max(log_prob_positive, log_prob_negative)
        
        prob_positive = math.exp(log_prob_positive - max_log_prob)
        prob_negative = math.exp(log_prob_negative - max_log_prob)
        
        # Normalize probabilities so they sum to 1
        # This gives us proper probability distributions
        total_prob = prob_positive + prob_negative
        prob_positive /= total_prob
        prob_negative /= total_prob
        
        # Make final prediction: choose the class with higher probability
        predicted_class = 'positive' if prob_positive > prob_negative else 'negative'
        
        return predicted_class, prob_positive, prob_negative
    
    def evaluate(self, test_pos_dir: str, test_neg_dir: str) -> Dict[str, float]:
        """
        Evaluate the classifier on test data.
        
        This method performs comprehensive evaluation of the trained classifier
        on unseen test data. It calculates accuracy, precision, recall, and F1-score
        for both positive and negative classes, providing a complete performance assessment.
        
        The evaluation process:
        1. Loads test data from positive and negative directories
        2. Makes predictions on all test reviews
        3. Builds confusion matrix (TP, FP, TN, FN)
        4. Calculates performance metrics for each class
        5. Returns comprehensive results dictionary
        
        Args:
            test_pos_dir (str): Directory containing positive test reviews
            test_neg_dir (str): Directory containing negative test reviews
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics:
                - accuracy: Overall classification accuracy
                - precision_positive/negative: Precision for each class
                - recall_positive/negative: Recall for each class
                - f1_positive/negative: F1-score for each class
                - true_positives/negatives: Count of correct predictions
                - false_positives/negatives: Count of incorrect predictions
        """
        print("Loading test data...")
        
        # Load test reviews from both positive and negative directories
        pos_reviews = self.load_reviews(test_pos_dir, 'positive')
        neg_reviews = self.load_reviews(test_neg_dir, 'negative')
        
        # Initialize counters for evaluation metrics
        correct_predictions = 0  # Total correct predictions
        total_predictions = 0    # Total predictions made
        
        # Confusion matrix components
        true_positives = 0   # Correctly predicted positive reviews
        false_positives = 0  # Incorrectly predicted as positive
        true_negatives = 0   # Correctly predicted negative reviews
        false_negatives = 0  # Incorrectly predicted as negative
        
        # Evaluate on positive test reviews
        print("Evaluating on positive test reviews...")
        for review in pos_reviews:
            predicted, _, _ = self.predict_single(review)
            total_predictions += 1
            
            if predicted == 'positive':
                correct_predictions += 1
                true_positives += 1
            else:
                false_negatives += 1  # Missed positive review (predicted as negative)
        
        # Evaluate on negative test reviews
        print("Evaluating on negative test reviews...")
        for review in neg_reviews:
            predicted, _, _ = self.predict_single(review)
            total_predictions += 1
            
            if predicted == 'negative':
                correct_predictions += 1
                true_negatives += 1
            else:
                false_positives += 1  # Incorrectly predicted as positive
        
        # Calculate overall accuracy
        accuracy = correct_predictions / total_predictions
        
        # Calculate precision, recall, and F1-score for positive class
        # Precision = TP / (TP + FP) - Of all positive predictions, how many were correct?
        precision_positive = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        # Recall = TP / (TP + FN) - Of all actual positives, how many did we catch?
        recall_positive = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        # F1-score = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean
        f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive) if (precision_positive + recall_positive) > 0 else 0
        
        # Calculate precision, recall, and F1-score for negative class
        precision_negative = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        recall_negative = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative) if (precision_negative + recall_negative) > 0 else 0
        
        # Compile all results into a comprehensive dictionary
        results = {
            'accuracy': accuracy,
            'precision_positive': precision_positive,
            'recall_positive': recall_positive,
            'f1_positive': f1_positive,
            'precision_negative': precision_negative,
            'recall_negative': recall_negative,
            'f1_negative': f1_negative,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
        
        return results
    
    def predict_review(self, review_text: str) -> None:
        """
        Predict sentiment for a single review and display results.
        
        This is a convenience method that wraps the predict_single method
        and provides a user-friendly display of the prediction results.
        It shows the review text (truncated), predicted sentiment, and
        confidence scores in a readable format.
        
        Args:
            review_text (str): The movie review text to classify and display
            
        Note:
            - Review text is truncated to 200 characters for display
            - Shows both positive and negative probabilities
            - Displays overall confidence level (max of the two probabilities)
        """
        # Get prediction results from the core prediction method
        predicted, pos_prob, neg_prob = self.predict_single(review_text)
        
        # Display the review text (truncated for readability)
        print(f"\nReview: {review_text[:200]}{'...' if len(review_text) > 200 else ''}")
        
        # Display the predicted sentiment in uppercase for emphasis
        print(f"Predicted sentiment: {predicted.upper()}")
        
        # Show confidence scores for both classes
        print(f"Confidence - Positive: {pos_prob:.4f}, Negative: {neg_prob:.4f}")
        
        # Display overall confidence level (how certain the model is)
        print(f"Confidence level: {max(pos_prob, neg_prob):.4f}")

def main():
    """
    Main function to train and evaluate the Naive Bayes classifier.
    
    This function orchestrates the complete workflow:
    1. Sets up file paths for the aclImdb dataset
    2. Validates that all required directories exist
    3. Creates and trains the Naive Bayes classifier
    4. Evaluates the classifier on test data
    5. Displays comprehensive performance metrics
    6. Provides an interactive mode for user testing
    
    The function expects the aclImdb dataset to be in the same directory
    as this script, with the following structure:
    - aclImdb/train/pos/ (positive training reviews)
    - aclImdb/train/neg/ (negative training reviews)
    - aclImdb/test/pos/ (positive test reviews)
    - aclImdb/test/neg/ (negative test reviews)
    """
    # Set up paths - get the directory where this script is located
    # This ensures the script works regardless of the current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "aclImdb")
    train_pos_dir = os.path.join(base_dir, "train", "pos")
    train_neg_dir = os.path.join(base_dir, "train", "neg")
    test_pos_dir = os.path.join(base_dir, "test", "pos")
    test_neg_dir = os.path.join(base_dir, "test", "neg")
    
    # Debug: Print the paths being checked for troubleshooting
    print(f"Script directory: {script_dir}")
    print(f"Base directory: {base_dir}")
    print(f"Train pos directory: {train_pos_dir}")
    print(f"Train neg directory: {train_neg_dir}")
    print(f"Test pos directory: {test_pos_dir}")
    print(f"Test neg directory: {test_neg_dir}")
    
    # Validate that all required directories exist before proceeding
    for directory in [train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir]:
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found!")
            print(f"Current working directory: {os.getcwd()}")
            return
    
    # Display program header
    print("=" * 60)
    print("MOVIE REVIEW SENTIMENT CLASSIFIER")
    print("Using Naive Bayes Algorithm")
    print("=" * 60)
    
    # Create and train the Naive Bayes classifier
    classifier = NaiveBayesClassifier()
    classifier.train(train_pos_dir, train_neg_dir)
    
    # Display evaluation section header
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Evaluate the trained classifier on test data
    results = classifier.evaluate(test_pos_dir, test_neg_dir)
    
    # Display comprehensive performance metrics
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Display metrics for positive class
    print(f"\nPositive Class Metrics:")
    print(f"  Precision: {results['precision_positive']:.4f}")
    print(f"  Recall: {results['recall_positive']:.4f}")
    print(f"  F1-Score: {results['f1_positive']:.4f}")
    
    # Display metrics for negative class
    print(f"\nNegative Class Metrics:")
    print(f"  Precision: {results['precision_negative']:.4f}")
    print(f"  Recall: {results['recall_negative']:.4f}")
    print(f"  F1-Score: {results['f1_negative']:.4f}")
    
    # Display confusion matrix for detailed analysis
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {results['true_positives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  True Negatives: {results['true_negatives']}")
    print(f"  False Negatives: {results['false_negatives']}")
    
    # Interactive prediction mode for user testing
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION")
    print("=" * 60)
    print("Enter your own movie review to test the classifier!")
    print("Type 'quit' to exit.\n")
    
    # Interactive loop for user input
    while True:
        user_review = input("Enter a movie review (quit to quit): ").strip()
        
        # Check for exit command
        if user_review.lower() == 'quit':
            break
        
        # Process non-empty reviews
        if user_review:
            classifier.predict_review(user_review)
            print()  # Add spacing between predictions

if __name__ == "__main__":
    main()
