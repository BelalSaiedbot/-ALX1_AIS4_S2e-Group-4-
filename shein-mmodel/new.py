import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import json
import re

# Set your API key directly
os.environ["API_KEY"] = "AIzaSyCYoyXHPR6s7mtIYkz5aJYfaa5mPLrZflA"
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Load schema for Gemini model
with open("./scheme.json", "r") as f:
    gemini_flash_schema = json.load(f)

# Preprocess text function
def preprocess_text(text):
    stopwords = {
        "the", "is", "in", "at", "on", "a", "an", "and", "or", "for", "to", "of", "with", "that", "by", "it",
    }
    text = re.sub(r"\d+|[^\w\s]|\s+", " ", text.lower()).strip()
    return " ".join([word for word in text.split() if word not in stopwords])

# Classify reviews as 'fake' or 'real'
def classify_reviews(product_review_text):
    prompt = (
        f"Here is a product review: '{product_review_text}'"
        " Classify this review as either 'fake' or 'real' based on its content."
        " If the review contains fewer than 3 words, classify it as 'fake'."
        " Respond only with 'fake' or 'real'."
    )
    try:
        response = model.generate_content(prompt)
        classification = response.text.strip().lower()
        return classification
    except Exception as e:
        st.error(f"Error classifying review: {product_review_text}. Error: {e}")
        return "error"

# Read CSV with different encodings
def read_csv_with_encodings(file_path):
    encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError as e:
            st.warning(f"Error reading with encoding {enc}: {e}")
            continue
    st.error("Failed to read the CSV file with all tried encodings.")
    return None

# Generate sentiment and grade using Gemini
def generate_review_grade_with_sentiment(review_text):
    try:
        prompt = f"Analyze the following review, determine its sentiment (positive, negative, neutral), and rate it from 1 to 5: {review_text}"
        response = model.generate_content(prompt)

        sentiment_match = re.search(r"(positive|negative|neutral)", response.text, re.IGNORECASE)
        grade_match = re.search(r"\d(\.\d+)?", response.text)

        if sentiment_match and grade_match:
            sentiment_label = sentiment_match.group().upper()
            grade = float(grade_match.group())
            return sentiment_label, grade
        else:
            st.write(f"No valid sentiment or grade found in response: {response.text}")
            return None, None
    except Exception as e:
        st.error(f"Error generating sentiment and grade for review: {e}")
        return None, None

# Generate summary using Gemini
def generate_summary(text):
    try:
        schema_str = json.dumps(gemini_flash_schema)
        prompt = f"Using the following constraints: {schema_str}, summarize the following text: {text}"
        response = model.generate_content(prompt)
        summary = response.text.strip()
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Summary could not be generated."

# Generate pros and cons using Gemini
def generate_pros_and_cons(reviews_text):
    try:
        schema_str = json.dumps(gemini_flash_schema)
        prompt = f"Using the following constraints: {schema_str}, extract pros and cons from the following text: {reviews_text}"
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        pros, cons = "", ""
        
        # Extract pros and cons from the response
        if "Pros:" in response_text and "Cons:" in response_text:
            pros = response_text.split("Pros:")[1].split("Cons:")[0].strip()
            cons = response_text.split("Cons:")[1].strip()
        elif "Pros:" in response_text:
            pros = response_text.split("Pros:")[1].strip()
        elif "Cons:" in response_text:
            cons = response_text.split("Cons:")[1].strip()

        return pros, cons
    except Exception as e:
        st.error(f"Error generating pros and cons for the reviews: {e}")
        return "Pros could not be generated.", "Cons could not be generated."

# Process product reviews
def process_product_reviews(product_reviews):
    classifications = []
    for review in product_reviews["product_review_name"]:
        classification = classify_reviews(review)
        classifications.append(classification)
    product_reviews['classification'] = classifications

    total_reviews = len(product_reviews)
    avg_product_rating = product_reviews["product_rating"].mean() if not product_reviews["product_rating"].isnull().all() else 0
    real_reviews_count = len(product_reviews[product_reviews['classification'] == 'real'])

    results = {
        "Total Reviews": total_reviews,
        "Average Product Rating": avg_product_rating,
        "Real Reviews Count": real_reviews_count,
    }
    
    return results

# Save results to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        st.success(f"Results saved to {filename}")
    except Exception as e:
        st.error(f"Error saving to JSON: {e}")

# Streamlit App Layout
st.title("Product Review Analyzer and Grader")

df = read_csv_with_encodings("./newData.csv")

if df is not None:
    product_id = st.text_input("Enter Product ID:")
    
    if product_id:
        st.session_state.results = {}

        st.subheader(f"Reviews for Product ID '{product_id}'")
        filtered_reviews = df[df["product_id"].astype(str) == product_id]

        if not filtered_reviews.empty:
            # Combine all reviews for the product
            combined_reviews_text = " ".join(filtered_reviews["product_review_name"].tolist())

            # Overall classification
            st.subheader("Overall Classification of Reviews")
            overall_classification = classify_reviews(combined_reviews_text)
            st.write(f"The reviews for the product are classified as '{overall_classification}'.")

            if overall_classification == 'real':
                # Summarization
                st.subheader("Summarization")
                summary = generate_summary(combined_reviews_text)
                st.write(f"Summary: {summary}")

                # Generate pros and cons
                pros, cons = generate_pros_and_cons(combined_reviews_text)
                st.write(f"Pros: {pros}\nCons: {cons}")

                # Grades and Ratings
                st.subheader("Grades and Ratings")
                result = process_product_reviews(filtered_reviews)
                st.write(
                    f"Total Reviews: {result['Total Reviews']}\n"
                    f"Average Product Rating: {result['Average Product Rating']}\n"
                    f"Real Reviews Count: {result['Real Reviews Count']}"
                )

                # Save results to JSON
                output_data = {
                    "Product ID": product_id,
                    "Overall Classification": overall_classification,
                    "Summary": summary,
                    "Pros": pros,
                    "Cons": cons,
                    "Grades and Ratings": result
                }
                save_to_json(output_data, f"{product_id}_review_analysis.json")
        else:
            st.error(f"No reviews found for Product ID: {product_id}")
