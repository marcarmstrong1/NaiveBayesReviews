This is going to be a project that takes online reviews about video games and tries to figure out what people like about certain video games and what they dislike about them.

Ultimately the goal is to create a classifier to run though the reviews and decide if the reviewer actually liked the game or not. Then to summarize what was liked and what wasn't liked.

The steps I took:

1. Find data on reviews: Done
2. Tokenize using TF-IDF: Done
3. Build NB Classifier: Done
4. Test Accuracy of model (Simple Lexicons get 40% and SOTA on HuggingFace gets 60-70%): Done
5. Summarize findings of likes and dislikes to recommend to team: Not going to do

The project scope is simple but should showcase the effectiveness of simple models for sentiment. My thought is that the words used in 4&5 star reviews is significantly different than 1&2 stars.

Project is complete. The overall accuracy for the model is 0.79 to compare with simple Lexicon Dictionary model its almost twice as good. Also, State of the Art (SOTA) models on Huggingface achieve scores of 60-70% accuracy. 

Obviously with the advent of powerful LLM's such as OpenAI's ChatGPT and Google's PaLM 2 models, this project could've been trivialized with a prompt like "Read through the review and pick out what features of the video game the reviewer liked." Then other analytics could be run to find similar themes amongst the responses and relayed to the marketing team. However, those LLM's usualy cost moeny for API calls so for a simple task like this it could be achieved in-house for essentially free.

Even further, one could try to build a deep neural net with LSTM and see what accuracy they could achieve but the training of that model given the dataset size of almost 500k reviews would take longer to build the model. One could also try to use fancy machine learning models like XGBoost but that again my be overkill. Given that just simply checking the Emotional Lexicons gives an almost 50% accuracy it seems that there is a decent split in the type of vocabulary used in positive and negative reviews. Which is why I thought the Naive Bayes model was best suited for this and it would appear to be out even a few SOTA models on HuggingFace.

Never doubt the simple solution! It took less time to build and run than training many models would and if this was a time sensitive business issue you could have actionable insights faster. 