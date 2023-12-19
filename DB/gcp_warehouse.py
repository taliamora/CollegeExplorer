
from google.cloud import bigquery
from google.oauth2 import service_account
import uuid
import numpy as np
from RateMyProfessor.rmp_api import get_schools_reviews, get_uni_by_name
from DB.DataPrep import PreProcess, prep_data
from openai import OpenAI
from tensorflow.keras.models import load_model
import pickle 
import os
import traceback
from google.cloud.exceptions import NotFound
from transformers import GPT2Tokenizer




# Load Models
Tokenizer = pickle.load(open("SentimentAnalysis/SentimentAnalysisTokenizer.pkl", 'rb'))
SentimentModel = load_model("SentimentAnalysis/best_model.h5")


def fetch_summaries(school, OpenAI_client, GCP_client):
        query = f"""
        SELECT positive_summary, negative_summary
        FROM `college-explorer-app.RateMyProfessor.summaries_table`
        WHERE school_name = @school_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("school_name", "STRING", school)])

        try:
            # First, try to fetch summaries from the warehouse
            query_job = GCP_client.query(query, job_config=job_config)
            results = query_job.result()

            # Check if any rows are returned
            if results.total_rows == 0:
                print(f"No summaries found for {school}. Generating summaries...")

                # Get reviews & generate summaries, then populate the warehouse
                message, action = Summarize(school, OpenAI_client, GCP_client)
                if not action:
                    return message, None, None
                elif action:
                    # Try fetching again
                    query_job = GCP_client.query(query, job_config=job_config)
                    results = query_job.result()
            for row in results:
                return None, row.positive_summary, row.negative_summary
        except NotFound:
            print("The specified table was not found in BigQuery.")
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()  # Print full stack trace



def Summarize(school, OpenAI_client, GCP_client):
    summary_id = make_uuid()
    uni = get_uni_by_name(school)
    if not uni:
        message = f"Unfortunately, {school} doesn't seem to be in the reviews database 😢"
        action = None

    else:
    # Get Reviews & Split on Sentiment
        positive, negative = classify_reviews(str(uni), summary_id, GCP_client)

    # Feed to summarizer
        if positive == []:
            positive = negative
        elif negative == []:
            negative = positive
        pos_sum = gpt_summary(positive, "positive", OpenAI_client)
        neg_sum = gpt_summary(negative, "negative", OpenAI_client)
        
        # Load Data to Warehouse
        summary_data = {
            "school_name": school,
            "positive_summary": pos_sum,
            "negative_summary": neg_sum,
            "summary_id": summary_id  # Store summary ID
        }
        
        to_warehouse("RateMyProfessor", "summaries_table", [summary_data], GCP_client)
        message = f"{str(school)} complete."
        action = True
    return message, action


def make_uuid():
    return str(uuid.uuid4())


# Function to insert rows into BigQuery
def to_warehouse(dataset_name, table_name, rows_to_insert, GCP_client):
    
    table_id = f"{GCP_client.project}.{dataset_name}.{table_name}"
    errors = GCP_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
    
    


def gpt_summary(reviews, sentiment, OpenAI_client):
    ### FIRST: Ensure reviews are under token limit
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Convert the reviews list to a single string
    reviews_str = ' '.join(reviews)

    # Check the number of tokens
    tokens = tokenizer.encode(reviews_str, return_tensors='pt')
    print("\n\nGPT INPUT LENGTH:", tokens.shape[1])

    max_tokens = 3500

    # If the number of tokens exceeds the limit, truncate the reviews
    while tokens.shape[1] > max_tokens:
        # Remove the last review and re-check
        reviews.pop()
        reviews_str = ' '.join(reviews)
        tokens = tokenizer.encode(reviews_str, return_tensors='pt')

    response = OpenAI_client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {'role':'system', 'content': f"Summarize the list of reviews into a single paragraph that highlights the {sentiment} aspects of the school"},
            {'role': 'user', 'content': str(reviews)}
        ])
    return response.choices[0].message.content
    
    
    
    
def classify_reviews(school, summary_id, GCP_client):
    
    rev_df = get_schools_reviews(school, output="dataframe")
    
    rev_df = prep_data(rev_df, labels=False, save=False,)
    x_test = list(rev_df.Reviews)
    
    # Tokenize & Pad
    x_test_padded = PreProcess(x_test, Tokenizer)
    
    # Classify Reviews
    # Sentiment Analyses Model
    predictions = SentimentModel.predict([x_test_padded])
    # Get just the numeric label predictions for test data
    predictions[predictions > .5] = 1
    predictions[predictions <= .5] = 0
    predicted_labels = np.squeeze(predictions)

    zipped = zip(x_test, predicted_labels)
    reviews_data = []
    positive = []
    negative = []
    for rev, lab in zipped:
        # Prepare data for BigQuery
        reviews_data.append({"school_name": school,
                             "review_text": rev, 
                             "sentiment": int(lab), 
                             "summary_id": summary_id})
        if int(lab) == 1:
            positive.append(rev)
        elif int(lab) == 0:
            negative.append(rev)

    to_warehouse("RateMyProfessor", "reviews_table", reviews_data, GCP_client)
    return positive, negative
    
    
    

        
        
        


        