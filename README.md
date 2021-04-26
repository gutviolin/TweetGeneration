# TweetGeneration
A GPT-2 implementation which uses a custom dataset of tweets, gathered with the twitter API.

# Encode the tweets using the vocabulary for the 124M model.
python encode.py tweets.txt tweets.npz --model_name 124M

# Re-train the model on the dataset.
python train.py --dataset tweets.npz --model_name 124M

# Generate unconditional samples (if model has been retrained you must copy the new model top the Tweeter folder)
python generate_unconditional_samples.py --model_name Tweeter

# Generate unconditional samples (if model has been retrained you must copy the new model top the Tweeter folder)
python interactive_conditional_samples.py --model_name Tweeter
