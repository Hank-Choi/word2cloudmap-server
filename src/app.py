from flask import Flask, jsonify, request

from src.token_generator import get_random_string
from src.word2vec import train, preprocess, search
import uuid
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
app.config["FLASK_DEBUG"] = False


# query_params = request.values[0]
# body_form_data = request.values[1]
# body_raw_json = request.json

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/model/', methods=['POST'])
@cross_origin()
def post_model():
    print("hi")
    if request.method == 'POST':
        print(request.values)
        # body_form_data = request.values['paragraph']
        paragraph = request.values['paragraph']

        # paragraph = body_form_data['paragraph']
        unique_id = str(uuid.uuid1())

        rank = train(preprocess(paragraph), unique_id)
        print(rank)
        response = {
            'unique_id':unique_id,
            'rank':rank
        }
        return response


@app.route('/words/', methods=['GET'])
def get_similar_words():
    if request.method == 'GET':
        query_params = request.values
        words_str = query_params['base_word']
        print(words_str)

        if words_str:
            # words = words_str.split(',')
            # search(request.headers['uuid'], words)
            words = search(request.headers['uuid'], words_str)
            return {
                'words': words
            }

        # convert that to bytes
    return 'hello'


if __name__ == '__main__':
    app.run()
