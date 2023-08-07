from flask import Flask, render_template, request
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np



app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = ''


# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    message = request.json.get("message")

    df = pd.read_csv("testWithEmbeddings.csv")
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    most_similar_entry = search(df, message, n=1)
    similarity_score = most_similar_entry.iloc[0]['similarities']
    response = ""

    if similarity_score < .8:
        response = "I am unable to generate a proper response at this time."
    else:
        prompt = f"{message} [SEP] {most_similar_entry.iloc[0]['Answer']}"
        response = get_chat_completion(prompt).content

    res = {}
    res["content"] = response
    return res



# Get embeddings for each question in csv
def run_embedding_model():
    embedding_model = "text-embedding-ada-002"
    df = pd.read_csv("test.csv")  # csv reader

    # Concatenate question and answer before passing to the embedding model
    df["QuestionWithAnswer"] = (
    "Question: " + df.Question.str.strip() + "; Answer: " + df.Answer.str.strip()
    )

    df["embedding"] = df.QuestionWithAnswer.apply(lambda x: get_embedding(x, engine=embedding_model))  # runs each question + answer in the CSV through embedding model
    df.to_csv("testWithEmbeddings.csv", index=False)  # writes results to a new csv file. This csv file now includes each question's embedding

# Compare user's question to question embeddings
def search(df, userQuestion, n=3):
   embedding = get_embedding(userQuestion, 'text-embedding-ada-002') # gets embedding for user question
   df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, embedding)) # creates a new csv entry with a similarity score
   res = df.sort_values('similarities', ascending=False).head(n)
   first_answer = res # retrieve the first answer from the 'Answer' column
   return first_answer # return the first answer as a string


def get_chat_completion(message):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "Keep the answer brief."},
            {"role": "user", "content": message}    
        ]
    )
    if completion.choices[0].message!=None:
        return completion.choices[0].message
    else :
        return 'Failed to Generate response!'
    

if __name__=='__main__':
    app.run()
    

