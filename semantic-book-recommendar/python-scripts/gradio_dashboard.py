import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
import seaborn as sns
import io
import gradio as gr
from PIL import Image

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books ["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books ["large_thumbnail"],
)

raw_documents = TextLoader ("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [
        int(rec[0].page_content.strip('"').split()[0]) if isinstance(rec, tuple)
        else int(rec.page_content.strip('"').split()[0])
        for rec in recs]    
    book_recs = books.loc[books["isbn13"].isin(books_list)]
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Tone-based emotion sorting
    if tone and tone != "All":
        tone_column_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }
        tone_col = tone_column_map.get(tone)
        if tone_col and tone_col in book_recs.columns:
            book_recs = book_recs.sort_values(by=tone_col, ascending=False)

    return book_recs.head(final_top_k)

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows ():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results

# --- Functions for Visuals and Stats --- #
def plot_pie(column):
    fig, ax = plt.subplots(figsize=(6, 6))
    books[column].fillna("Unknown").value_counts().head(5).plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title(f"Top 5 {column} Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return Image.open(buf)

def get_missing_df():
    return books.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"})

def get_summary_df():
    return books.describe(include="all").T.fillna("").reset_index().rename(columns={"index": "Column"})

def filter_by_rating(min_rating):
    filtered = books[books["average_rating"] >= min_rating]
    return filtered[["title", "average_rating", "authors"]].head(20)

def plot_author_boxplot():
    fig, ax = plt.subplots(figsize=(8, 4))
    author_counts = books["simple_categories"].value_counts().index[:5]
    data = books[books["simple_categories"].isin(author_counts)]
    data["num_authors"] = data["authors"].fillna("").apply(lambda x: len(str(x).split(";")))
    sns.boxplot(data=data, x="simple_categories", y="num_authors", hue="simple_categories", palette="Set2", ax=ax, legend=False)
    ax.set_title("Number of Authors per Category")
    ax.set_ylabel("Number of Authors")
    ax.set_xlabel("Category")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return Image.open(buf)

def get_thumbnails(category):
    df = books[books["simple_categories"] == category].dropna(subset=["thumbnail"]).head(8)
    return list(df["thumbnail"])

# Category & tone setup
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom theme
custom_theme = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="stone",
    font=["Plus-jakarta-sans", "sans-serif"]
)

# Gradio UI
with gr.Blocks(theme=custom_theme) as dashboard:
    with gr.Tab("🔍 Recommender"):
        gr.Markdown("""
        <style>
        #form-section {
            padding: 18px;
            border: 1px solid #dcdcdc;
            border-radius: 15px;
            background: #fdfdfd;
            margin-bottom: 1.5rem;
        }
        .title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1em;
        }
        </style>
        """)

        gr.Markdown("# 📚 Semantic Book Recommender", elem_classes="title")
        gr.Markdown("Describe your ideal book and get smart recommendations based on semantics and emotions 🎯")

        with gr.Group(elem_id="form-section"):
            with gr.Row():
                user_query = gr.Textbox(
                    label="🔍 Book Description",
                    placeholder="e.g., A story about forgiveness, mystery, and redemption",
                    lines=2
                )
            with gr.Row():
                category_dropdown = gr.Dropdown(choices=categories, label="📂 Category", value="All")
                tone_dropdown = gr.Dropdown(choices=tones, label="🎭 Emotional Tone", value="All")
            with gr.Row():
                submit_button = gr.Button("🚀 Find Recommendations", variant="primary")

        gr.Markdown("## 🧠 Smart Recommendations")
        output = gr.Gallery(label="📚 Recommended Books", columns=4, rows=2, height="auto", preview=False)

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output
        )

    with gr.Tab("📊 Dataset Statistics"):
        gr.Markdown("## 🧮 Dataset Summary Table")
        gr.Dataframe(value=get_summary_df(), interactive=False)

        gr.Markdown("## ❓ Missing Values Table")
        gr.Dataframe(value=get_missing_df(), interactive=False)

        gr.Markdown("## 🧁 Pie Chart Visualization")
        categorical_cols = books.select_dtypes(include=["object", "category"]).columns.tolist()
        col_dropdown = gr.Dropdown(
            choices=categorical_cols,
            value=categorical_cols[0] if categorical_cols else None,
            label="Select Column"
        )
        pie_img = gr.Image(type="pil", label="Pie Chart")
        col_dropdown.change(fn=plot_pie, inputs=col_dropdown, outputs=pie_img)

        gr.Markdown("## 🌡️ Histogram Filter by Rating")
        rating_slider = gr.Slider(minimum=0, maximum=5, step=0.1, value=3.5, label="Minimum Rating")
        rating_table = gr.Dataframe(label="Books Above Rating", interactive=False)
        rating_slider.change(fn=filter_by_rating, inputs=rating_slider, outputs=rating_table)

        gr.Markdown("## 📦 Boxplot: Authors per Category")
        box_img = gr.Image(type="pil", value=plot_author_boxplot, label="Author Count Boxplot")

        gr.Markdown("## 🖼️ Top Book Covers by Category")
        cat_dropdown = gr.Dropdown(choices=books["simple_categories"].dropna().unique().tolist(), label="Select Category")
        gallery = gr.Gallery(label="Thumbnails", columns=4, height="auto")
        cat_dropdown.change(fn=get_thumbnails, inputs=cat_dropdown, outputs=gallery)


# Run app
if __name__ == "__main__":
    dashboard.launch()

