This is going to be a project that takes online reviews about video games and tries to figure out what people like about certain video games and what they dislike about them.

Ultimately the goal is to create a classifier to run though the reviews and decide if the reviewer actually liked the game or not. Then to summarize what was liked and what wasn't liked.

The steps I intend to take:

1. Find data on reviews: Done
2. Tokenize using TF-IDF: Done
3. Build NB Classifier: Done
4. Test Accuracy of model (Simple Lexicons get 40% and SOTA on HuggingFace gets 60-70%): In Progress 
5. Summarize findings of likes and dislikes (maybe n-gram?) to recommend to team: NOT Started

The project scope is simple but should showcase the effectiveness of simple models for sentiment. My thought is that the words used in 4&5 star reviews is significantly different than 1&2 stars.

Currently, need to fix review data to be string format to run classifier. Also need to remove NaN from review data.

I am thinking that a simple model can be as effective as SOTA from Huggingface.
