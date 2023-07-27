from flask import Flask
from flask import request, jsonify, Response
import sys
import pandas as pd
import io

sys.path.append("..")
from deployment import StampClassifier

#  Response,stream_with_context,request,Blueprint,current_app,g,jsonify


app = Flask(__name__)
app.clf = StampClassifier()


@app.route("/")
def index():
    return "Welcome to ALERCE One Stamp Classifier service"


@app.route("/get_classification", methods=("POST",))
def get_classification():
    _ = request.get_json()
    files = request.files

    f = {}
    for key in files:
        f[key] = files[key].read()

    l = [(f["cutoutScience"], f["cutoutTemplate"], f["cutoutDifference"])]
    l_df = pd.DataFrame(
        l, columns=["cutoutScience", "cutoutTemplate", "cutoutDifference"]
    )
    metadata = pd.read_csv(io.BytesIO(f["metadata"]))
    l_df = pd.concat([l_df, metadata], axis=1)

    try:
        pred = app.clf.execute(l_df)
        pred = pred.iloc[0].to_dict()
    except OverflowError:
        return Response(
            '{"status":"ERROR", "content": "Stamp files has too many NaN values"}', 500
        )
    except ValueError:
        return Response(
            '{"status":"ERROR", "content": "Input image is not squared"}', 500
        )

    response = dict()
    response["status"] = "SUCCESS"
    response["probabilities"] = pred
    return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
