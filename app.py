import os

from flask import Flask, render_template, request
import openai

app = Flask(__name__)

#openai tools fine_tunes.prepare_data -f .\chatbotEnvironment\chatbotTrain.jsonl
#openai --api-key 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 api fine_tunes.create -t .\chatbotEnvironment\chatbotTrain_prepared.jsonl -m ada
#openai --api-key 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 api fine_tunes.list
@app.route("/", methods = ["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = openai.Completion.create(
            model = "ada:ft-personal-2023-05-01-09-12-33",
            prompt = prompt,
            n = 1,
            max_tokens = 1024,
            stop = None,
        )
        return render_template("home.html", response = response.choices[0])
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug = True)
