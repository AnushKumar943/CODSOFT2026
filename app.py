from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def index():
    result = ""

    if request.method == "POST":
        values = [float(x) for x in request.form.values()]
        final = [np.array(values)]

        pred = model.predict(final)[0]

        if pred == 1:
            result = "⚠ Customer Likely to Churn"
        else:
            result = "✅ Customer Will Stay"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
