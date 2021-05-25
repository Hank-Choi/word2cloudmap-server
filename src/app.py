from flask import Flask, jsonify, request

app = Flask(__name__)
app.config["FLASK_DEBUG"] = True


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        paragraph = request.json
        # convert that to bytes
        # img_bytes = file.stream.read()
        # class_id, class_name = get_prediction(image_bytes=img_bytes)
        # return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
