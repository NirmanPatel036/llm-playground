import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm

books = pd. read_csv("books_with_categories.csv")

# test the HF model with sample-text
classifier = pipeline("text-classification",
					  model="j-hartmann/emotion-english-distilroberta-base", 
					  top_k=None,
					  device='mps')

"""result = classifier("I love this!")

# as it returns a list of dictionaries
for item in result[0]:
	print(item)
    #another way, print(f"{item['label']:<10}: {item['score']}")"""

# break the description into sentences and give individual scores for each
"""sentences = books["description"][0].split(".")
predictions = classifier(sentences)

i = len(predictions)
while i > 0:
	for sentence in predictions[i-1]:
		print(sentence)
	print("--------------------")
	i -= 1"""

# create a dictionary with emotions having maximum probabilities from each sentence
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

def calculate_max_emotion_scores(predictions):
	per_emotion_scores = {label: [] for label in emotion_labels}
	for prediction in predictions:
		sorted_predictions = sorted(prediction, key=lambda x: ["label"])
		for index, label in enumerate (emotion_labels):
			per_emotion_scores[label].append(sorted_predictions[index]["score"])
	return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

# for all the books in dataset
for i in tqdm(range(len(books))):
	isbn.append(books["isbn13"][i])
	sentences = books["description"][i].split(".")
	predictions = classifier(sentences)
	max_scores = calculate_max_emotion_scores(predictions)
	for label in emotion_labels:
		emotion_scores[label].append(max_scores[label])

# create a new dataframe from the results
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn
print(emotions_df)

books = pd.merge(books, emotions_df, on = "isbn13")
books.to_csv("books_with_emotions.csv", index = False)

