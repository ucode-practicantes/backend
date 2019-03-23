from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello world"

@app.route('/upload', methods=["POST"])
def upload_data():
    """
    data: {
        ...
    },
    position: {
        x: 1,
        y: 1
    },
    id: "mac del device"
    """
    message = request.json.get("msg", "")
    return jsonify(dict(message=message.upper()))

if __name__ == "__main__":
    app.run()