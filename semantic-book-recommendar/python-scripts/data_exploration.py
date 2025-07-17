import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#------------------------ data cleaning ---------------------------#

books = pd.read_csv('books.csv')
print("Initial look at the dataset:\n", books.head())

#plot showing the missing values from the dataset
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel ("Columns")
plt.ylabel ("Missing values")
plt.show()

#marking if the description is present or not by adding the below column: 1 where it's missing else 0
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2025 - books["published_year"]

columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]

correlation_matrix = books[columns_of_interest].corr(method='spearman')

sns.set_theme (style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman correlation"})
heatmap.set_title("Correlation heatmap")
plt.show()

#checking all the books where anyone of the columns_of_interest data is missing
print(books[(books["description"].isna()) |
			(books ["num_pages"]. isna()) |
			(books["average_rating"].isna()) |
			(books["published_year"]. isna())
])

#hence, removing all such books from the dataframe
books_updated = books[~(books["description"].isna()) &
					~(books ["num_pages"]. isna()) &
					~(books["average_rating"].isna()) &
					~(books["published_year"]. isna())
]

'''
Potential problems for not making a copy of modified DataFrame: 

Unpredictable behavior - Code might work sometimes and fail other times
Silent data corruption - Changes might affect the original DataFrame unexpectedly
Hard-to-debug issues - The behavior depends on pandas' internal optimizations, which can change
Future pandas versions - This might become an error instead of just a warning
'''
books_updated = books_updated.copy()
print("Removing the above observations..\nUpdated dataset:\n", books_updated)

#just for visualization - first 100 entries
visualize_top = books_updated[:100]

#take a closer look at the distribution of the categories in the descending order
print("Categorical Distribution:\n", books_updated["categories"].value_counts().reset_index().sort_values("count", ascending=False))
sns.histplot(data=visualize_top, x = 'categories', kde=True)
plt.xticks(rotation=90)
plt.show()

#taking a closer look at the first few descriptions
print(books["description"].head())

'''IMP Note: Books with one-word descriptions won't enable a smooth recommendation process. They wouldn't be useful in the process
			 Hence, it makes sense to remove such observations from the dataset.'''

#introduce a new variable - gives us the length of the description
books_updated["words_in_description"] = books_updated["description"].str.split().str.len()
print(books_updated["words_in_description"].head())
sns.histplot(data=books_updated, x = 'words_in_description', kde=True)
plt.xticks(rotation=45)
plt.show()

'''
#books with description words ranging from 1 to 4
print("Books having description of upto 4 words:\n",
	books_updated.loc[books_updated["words_in_description"].between(1, 4), "description"], "\n")

#books with description words ranging from 5 to 14
print("Books having description words ranging from 5 to 14:\n",
	books_updated.loc[books_updated["words_in_description"].between(5, 14), "description"], "\n")

#books with description words ranging from 15 to 24
print("Books having description words ranging from 15 to 24:\n",
	books_updated.loc[books_updated["words_in_description"].between(15, 24), "description"], "\n")

#books with description words ranging from 25 to 34
print("Books having description of upto 4 words:\n",
	books_updated.loc[books_updated["words_in_description"].between(25, 34), "description"],)
'''

#use 25 words in description as a cutoff and remove all with less than 25 words
books_updated_25_words = books_updated[books_updated["words_in_description"] >= 25]
#print("Books having description words more than 25:\n", books_updated_25_words)
books_updated_25_words = books_updated_25_words.copy()

#create a new column where if the subtitle is missing, append the the title itself in that field, or
#if both, title and subtitle are present, merge them as a string using a colon
books_updated_25_words["title_and_subtitle"] = ( 
	np.where(books_updated_25_words["subtitle"].isna(), books_updated_25_words["title"],
			 books_updated_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
)
#print("Title & Subtitle Column:\n", books_updated_25_words["title_and_subtitle"])


'''
create a new column with a tagged description. Why?
it's a good practice as compared to direct string matching and filtering while recommending bcz it can get messy and slow
isbn number is treated as an identifier and then later can be removed
'''
books_updated_25_words["tagged_description"] = books_updated_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
#print("Tagged Description Column:\n", books_updated_25_words["tagged_description"])

#now, removing all the unwanted columns and saving the dataframe as a new csv
(
	books_updated_25_words
	.drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
	.to_csv("books_cleaned.csv", index = False)
)

