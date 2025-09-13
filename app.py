from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import json
import zipfile
from adjust_odds import main
import io
import logging

app = Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see everything
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Enforce 0.5 defaults for features
def enforce_defaults(df):
    feature_cols = ["JockeyScore","HorseWeightScore","DrawScore", "Liability", "GoingScore","PaceScore"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5).replace(0.0, 0.5)
    if "Overround" in df.columns:
        df["Overround"] = pd.to_numeric(df["Overround"], errors="coerce").fillna("")
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.json
    w_market = float(data.get("w_market", 0.6))
    alpha = float(data.get("alpha", 0.97))
    shrink = float(data.get("shrink", 1.0))

    # Convert table JSON to DataFrame
    df = pd.DataFrame(data["table"])
    df = enforce_defaults(df)

    # Save CSVs
    input_csv = "input.csv"
    output_csv = "output.csv"
    df.to_csv(input_csv, index=False)

    # Run odds adjustment
    main(
        input_csv=input_csv,
        output_csv=output_csv,
        w_market=w_market,
        alpha=alpha,
        shrink=shrink,
        round_step=10,
        decimals=3
    )

    # Read processed output
    output_df = pd.read_csv(output_csv)

    # Return both input and output as JSON
    output = jsonify({
        "input": df.to_dict(orient="records"),
        "output": output_df.to_dict(orient="records")
    })

    logger.debug(f"Backend /process result input = {df.to_dict(orient='records')}")
    logger.debug(f"Backend /process result ouput = {output_df.to_dict(orient='records')}")

    return output

@app.route("/download", methods=["GET"])
def download_zip():
    input_csv = "input.csv"
    output_csv = "output.csv"

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.write(input_csv)
        zf.write(output_csv)
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="adjusted_odds.zip"
    )

if __name__ == "__main__":
    #app.run(debug=True)
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
