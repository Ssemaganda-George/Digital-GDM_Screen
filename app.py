from flask import Flask, request, jsonify, render_template, session
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = b'\xdb\x908\x9bsKD\x1c\x91\x8a\xd84\x01\xcb\xa5]\x8b\xa9n\x10\xd7\x1e\x11g'  # Use your generated secret key

# Load environment variables from .env file
load_dotenv()

# Load cleaned data
def load_cleaned_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Load embeddings from file
def load_embeddings(file_path):
    return np.load(file_path)

# Initialize global variables at the module level
data = load_cleaned_data('cleaned_data.txt')
embeddings = load_embeddings('embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieve context using cosine similarity
def retrieve_context(query, index, data, top_n=5):
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
    similarities = cosine_similarity(query_embedding, index)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    context_retrieval_time = time.time()
    print(f"Time for context retrieval: {context_retrieval_time - start_time} seconds")
    return [data[idx] for idx in top_indices]

# Generate response with OpenAI API
def generate_response_with_openai(conversation_history):
    start_time = time.time()
    openai.api_key = os.getenv('API_KEY')  # Load the API key from the environment variable

    # Format the conversation history for the OpenAI API
    messages = [{"role": "system", "content": "You are a helpful assistant specialized in health information, with a focus on gestational diabetes. Provide accurate, concise, and informative responses based on the given context. If the question is not related to health or gestational diabetes, politely inform the user that you can only provide information on health and gestational diabetes."}]
    for entry in conversation_history:
        if entry['query']:  # Ensure there's a query before adding
            messages.append({"role": "user", "content": entry['query']})
        if entry['response']:  # Ensure there's a response before adding
            messages.append({"role": "assistant", "content": entry['response']})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )

    answer = response.choices[0].message['content'].strip()
    response_generation_time = time.time()
    print(f"Time for response generation: {response_generation_time - start_time} seconds")
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    start_time = time.time()
    user_query = request.form['query']

    # Retrieve previous context from session
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # Generate context based on previous messages
    context = retrieve_context(user_query, embeddings, data)
    
    # Generate response based on the entire conversation history
    conversation_history = session['conversation_history']
    conversation_history.append({'query': user_query, 'response': ''})  # Add the new query to the history for context generation
    response = generate_response_with_openai(conversation_history)
    conversation_history[-1]['response'] = response  # Update the latest entry with the generated response

    # Update conversation history in the session
    session['conversation_history'] = conversation_history
    response_time = time.time()

    print(f"Time to get response from OpenAI: {response_time - start_time} seconds")
    print(f"Conversation History: {session['conversation_history']}")  # Debug statement to check conversation history
    
    return jsonify({'response': response, 'conversation_history': session['conversation_history']})

if __name__ == '__main__':
    app.run(debug=True)
