from flask import Flask, render_template, request, jsonify
import sys

sys.path.append('../app')

import classifier as cs

app = Flask(__name__)

# create classifier instance on start once
classifier = cs.Classifier(basedir="../")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def say_name():
    json = request.get_json()
    question = json['question']
    return jsonify(question=question, category=classifier.get_category(question))


if __name__ == '__main__':
    app.run(debug=True)
