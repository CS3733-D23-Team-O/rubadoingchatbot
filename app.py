import openai
from flask import Flask, request, render_template
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from langchain.chat_models import ChatOpenAI
import sys
import os

app = Flask(__name__)
API_KEY = os.environ.get("OPENAI_API_KEY")
# openai.api_key = "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"


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()
    print(directory_path)
    print(documents)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

# Define chatbot function
def ask_ai(prompt):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    print(f"Prompt: {prompt}")
    response = index.query(prompt)
    print(f"Response: {response}")
    if response is None:
        response = "I'm sorry, I do not understand"
    return response


# Define Flask routes
@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form.get('prompt', '').strip()
        if prompt:
            response = ask_ai(prompt)
        else:
            response = "Please enter a prompt"
        return render_template('home.html', response=response)
    return render_template('home.html')


# Start Flask application
if __name__ == '__main__':
    construct_index("chatbotEnvironment/")
    app.run(debug=True)
