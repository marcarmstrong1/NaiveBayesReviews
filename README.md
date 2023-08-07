This is going to be a project that takes online reviews about video games and tries to figure out what people like about certain video games and what they dislike about them.

Ultimately the goal is to create a classifier to run though the reviews and decide if the reviewer actually liked the game or not. Then to summarize what was liked and what wasn't liked.

The steps I intend to take:

1. Find data on reviews: Done
2. Tokenize using TF-IDF: Done
3. Build NB Classifier: Done
4. Test Accuracy of model (Simple Lexicons get 40% and SOTA on HuggingFace gets 60-70%): In Progress 
5. Summarize findings of likes and dislikes (maybe n-gram?) to recommend to team: NOT Started

The project scope is simple but should showcase some unique text analysis thinking. The project will become even simpler if all 5-star and 1-star reviews only talk about the game and not the expereince of buying. Likely will not be creating a neural net as it may be overkill. Might come back over top to create one. But likely not.

Currently, need to fix review data to be string format to run classifier. Also need to remove NaN from review data.
