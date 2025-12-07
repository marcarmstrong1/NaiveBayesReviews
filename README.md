# Amazon Video Game Review Classifier: The Efficiency of Naive Bayes

## Project Context

In the current landscape of AI, there is a temptation to solve every Natural Language Processing (NLP) problem with Large Language Models (LLMs) like GPT-4 or PaLM 2. While powerful, these models incur high inference costs and latency.

This project demonstrates that for specific business problems—such as sentiment classification—classic machine learning algorithms can outperform complex models in terms of speed and cost-efficiency while maintaining high accuracy.

## Project Overview

The goal of this project is to ingest raw Amazon video game reviews and classify the reviewer's sentiment. Instead of a simple binary (Good/Bad) classification, we map the star ratings to three sentiment categories:

  * **Negative:** 1 or 2 Stars (Mapped to 0)
  * **Neutral:** 3 Stars (Mapped to 1)
  * **Positive:** 4 or 5 Stars (Mapped to 2)

## The Baseline: Naive Bayes

We utilize a **Multinomial Naive Bayes** classifier. This algorithm relies on the probability of word frequencies (Bayes' Theorem).

[Image of Naive Bayes classifier concept for text analysis]

**Why this approach?**
The hypothesis is that the vocabulary used in 5-star reviews (e.g., "masterpiece," "addictive," "smooth") is fundamentally distinct from the vocabulary in 1-star reviews (e.g., "broken," "buggy," "refund"). Because the features (words) are distinct, a simple probabilistic model is highly effective.

### Pipeline Steps

1.  **Ingestion:** Reads line-delimited JSON data (`.json`).
2.  **Preprocessing:** Target variable generation based on star rating thresholds.
3.  **Vectorization:**
      * **CountVectorizer:** Converts text to a matrix of token counts.
      * **TF-IDF:** Transforms counts to frequency-weighted scores (penalizing words like "the" or "and" that appear everywhere).
4.  **Modeling:** Fits a Multinomial Naive Bayes model on the training set.

## Performance Analysis

This simple implementation achieves significant results:

  * **Lexicon Baselines:** Simple dictionary lookups typically achieve \~40% accuracy.
  * **HuggingFace SOTA:** Many complex, pre-trained Transformers achieve 60-70% on this specific noisy dataset without heavy fine-tuning.
  * **Our Naive Bayes Model:** Achieves **\~79% Accuracy**.

**Conclusion:** This model is computationally free compared to LLMs, trains in seconds rather than hours, and provides actionable insights immediately.

## Prerequisites

To run this script, you will need the following libraries:

```bash
pip install pandas scikit-learn numpy
```

## Data Requirements

The script expects a file named `Video_Games_5.json` in the root directory. This is a standard Amazon dataset (often available from UCSD datasets) formatted as line-delimited JSON.

## The Challenge for Interns

The current script is a pure classifier—it tells us *if* a user liked a game, but not *why*.

Your task is to extend this project to provide qualitative insights to a marketing team.

1.  **Feature Importance:** Extract the "top features" for each class from the Naive Bayes model. Which specific words are most predictive of a 1-star review vs. a 5-star review?
2.  **Model Comparison:** The documentation mentions XGBoost. Implement an XGBoost classifier on the TF-IDF vectors. Does the gradient boosting approach yield enough accuracy gain to justify the increased training time?
3.  **Topic Modeling (Advanced):** The original scope included summarizing "likes and dislikes." Since we are avoiding LLMs, implement **Latent Dirichlet Allocation (LDA)** to cluster reviews into topics (e.g., "Graphics," "Gameplay," "Story") and see if you can correlate specific topics with negative sentiment.

## Usage

Run the script to train the model and output the classification report:

```bash
python sentiment_model.py
```
