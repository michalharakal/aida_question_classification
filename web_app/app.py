from flask import Flask, render_template, request, jsonify
import sys

sys.path.append('../app')

import classifier as cs

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def say_name():
    json = request.get_json()
    question = json['question']
    return jsonify(question=question, category=cs.get_category(question))


if __name__ == '__main__':
    app.run(debug=True)
