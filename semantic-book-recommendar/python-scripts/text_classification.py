import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import os

books = pd.read_csv('books_cleaned.csv')

# print categories with more than 50 books
print(books["categories"].value_counts().reset_index().query("count > 50"))

# look at a specific category
print(books[books["categories"] == "Juvenile Fiction"])

# come up with a mapping to highlight subtle categories, rest will be NaN by default
category_mapping = {'Fiction' : "Fiction",
'Juvenile Fiction': "Children's Fiction",
'Biography & Autobiography': "Nonfiction",
'History': "Nonfiction",
'Literary Criticism': "Nonfiction",
'Philosophy': "Nonfiction",
'Religion': "Nonfiction",
'Comics & Graphic Novels': "Fiction",
'Drama': "Fiction",
'Juvenile Nonfiction': "Children's Nonfiction",
'Science': "Nonfiction",
'Poetry': "Fiction"}

# add the new column to the dataset
books["simple_categories"] = books["categories"].map(category_mapping)
print(books.head())

# applying the ZERO-SHOT CLASSIFICATION
# import the HF model
fiction_categories = ["Fiction", "Nonfiction"]
pipe = pipeline("zero-shot-classification",
				model="facebook/bart-large-mnli",
				device="mps"
)

# looking at the first known fiction entry in the dataframe
print(books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[0])

# set a sequence and categories to run/apply the classifier, check for the scpres once you run, the depic the probabilities
sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[0]
print(pipe(sequence, fiction_categories))
print("\n")

# get the predicted label from the output using post-processing
max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]
print("Predicted Label using ZERO-SHOT: " + max_label + "\n")

# define a function for generating predictions
def generate_predictions(sequence, categories):
	predictions = pipe(sequence, categories)
	max_index = np.argmax(predictions["scores"])
	max_label = predictions["labels"][max_index]
	return max_label

# take a sizeable sample of fiction and non-fiction, and prediict the label using the classifier
# compare it with the already known label
# the if-else is here to avoid re-generation during every run
if os.path.exists("predictions_results.csv"):
    predictions_df = pd.read_csv("predictions_results.csv")
else:
    # Run the loops, then save
    actual_cats = []
    predicted_cats = []

    for i in tqdm(range(0, 300)):
        sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
        predicted_cats += [generate_predictions(sequence, fiction_categories)]
        actual_cats += ["Fiction"]

    for i in tqdm(range(0, 300)):
        sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)[i]
        predicted_cats += [generate_predictions(sequence, fiction_categories)]
        actual_cats += ["Nonfiction"]

    predictions_df = pd.DataFrame({
        "actual_categories": actual_cats,
        "predicted_categories": predicted_cats
    })

    predictions_df.to_csv("predictions_results.csv", index=False)

# sets 1 for correct prediction and 0 for incorrect
predictions_df["correct_prediction"] = (
	np.where(predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0)
)
print(predictions_df)

# calculating the accuracy of our labelling
accuracy = predictions_df["correct_prediction"].sum()/len(predictions_df)
print("Labelling accuracy: ", accuracy*100, "%\n")

# use this model to identify the missing categories (kinda create a subset of the dataset and take ones only with missing categories)
isbns = []
predicted_cats = []

# consider the isbn13 so that we can merge back to the original dataframe
missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)

for i in tqdm(range(0, len(missing_cats))):
	sequence = missing_cats["description"][i]
	predicted_cats += [generate_predictions(sequence, fiction_categories)]
	isbns += [missing_cats["isbn13"][i]]

missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})
print("The missing predicted categories: \n")
print(missing_predicted_df)

# merge it back to the main dataframe
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"])
books = books.drop(columns = ["predicted_categories"])

# specific fiction categories with these keywords exclusively
print(books[books["categories"].str.lower().isin([
	"romance",
	"science fiction",
	"scifi",
	"fantasy",
	"horror",
	"mystery",
	"thriller",
	"comedy",
	"crime"
	"historical"
])])

# save the changes in a new file
books.to_csv("books_with_categories.csv", index=False)























