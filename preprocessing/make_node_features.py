import base64
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model from the Hugging Face model hub
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set max number of word-pieces that can be used for each sentence
model.max_seq_length = 512

# Select the dataset to use out of the following: p, ds, cs
dataset_id = "p"

data_dir = Path(f'../data/communication_networks/{dataset_id}')

users_df = pd.read_csv(data_dir / 'metadata/users.csv')
users_df['AboutMe'].fillna('', inplace=True)  # Replace NaN with empty string

questions_df = pd.read_csv(data_dir / 'metadata/questions.csv')
answers_df = pd.read_csv(data_dir / 'metadata/answers.csv')
comments_df = pd.read_csv(data_dir / 'metadata/comments.csv')

users_df['text'] = users_df['AboutMe'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
users_df['embeddings'] = [list(i) for i in
                          model.encode(users_df['text'], show_progress_bar=True, convert_to_numpy=True)]
users_df.to_csv(data_dir / 'metadata/users.csv', index=False)

questions_df['text1'] = questions_df['Title'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
questions_df['text2'] = questions_df['Body'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
questions_df['embeddings'] = [list(i) for i in
                              model.encode(questions_df['text1'] + ". " + questions_df['text2'],
                                           show_progress_bar=True, convert_to_numpy=True)]
questions_df.to_csv(data_dir / 'metadata/questions.csv', index=False)

answers_df['text'] = answers_df['Body'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
answers_df['embeddings'] = [list(i) for i in
                            model.encode(answers_df['text'], show_progress_bar=True, convert_to_numpy=True)]
answers_df.to_csv(data_dir / 'metadata/answers.csv', index=False)

comments_df['text'] = comments_df['Text'].apply(lambda x: base64.b64decode(x).decode('utf-8'))
comments_df['embeddings'] = [list(i) for i in
                             model.encode(comments_df['text'], show_progress_bar=True, convert_to_numpy=True)]
comments_df.to_csv(data_dir / 'metadata/comments.csv', index=False)
