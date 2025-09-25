# Movie Review Analyzer

A sentiment analysis project that uses a **Naive Bayes classifier (implemented from scratch)** to determine whether IMDB movie reviews are positive or negative.  

This project demonstrates applied machine learning concepts such as **text preprocessing, probability modeling, and evaluation metrics** without relying on prebuilt ML libraries.

---

## Tech Stack
- **Language**: Python  
- **Core Libraries**: 
  - `os`, `re`, `math`, `collections` (for data handling and text processing)  
  - No external ML libraries — Naive Bayes implemented manually  
- **Dataset**: [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
  - 25,000 training reviews (12,500 positive, 12,500 negative)  
  - 25,000 test reviews (12,500 positive, 12,500 negative)  

---

## Features
- **Naive Bayes Classifier from scratch**: implements Laplace smoothing and Bayes’ theorem.  
- **Text preprocessing**: lowercasing, cleaning, tokenization.  
- **Training pipeline**: builds vocabulary, calculates priors and conditional probabilities.  
- **Evaluation metrics**: accuracy, precision, recall, F1-score, and confusion matrix.  
- **Interactive mode**: input your own review and get predicted sentiment with confidence scores.  

---

## Results
- Achieved **82.29% accuracy** on the IMDB dataset.  
- Detailed metrics include precision, recall, F1-score, and class-specific performance.  

---
## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Movie-Review-Analyzer.git
cd Movie-Review-Analyzer
```

### 2. Run the Classifier
Make sure you have Python 3 installed, then run:
```bash
python MovieReviewer.py
```
