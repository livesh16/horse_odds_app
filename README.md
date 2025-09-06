# 🏇 Horse Racing Odds Adjuster

This project is a **horse racing odds adjustment tool**. It takes raw **baseline odds** and adjusts them using **features** (jockey, horse weight, draw, going, pace) and overrounds, producing a set of adjusted odds for betting purposes.

---

## Features

- **Input/Output:** CSV-based input and output. Download the results as a ZIP file.
- **Adjustments:** Uses horse and race features to adjust probabilities.
- **Overround Handling:** Applies bookmaker margin (overround) per race. If a race does not have a specified overround, the script defaults to using the sum of market-implied probabilities from the baseline odds.
- **Favorite-Longshot Skew:** Allows skewing probabilities to reflect favorite-longshot effects.
- **Customizable Parameters:** `w_market`, `alpha`, `shrink`, `round_step`, `decimals`, and feature weights (`betas`) are configurable.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

* `pandas`
* `numpy`
* `flask`

---

## Usage

### 1. Running the Flask App

Start the backend server:

```bash
python app.py
```

* The web interface will be available at `http://127.0.0.1:5000/`.
* Enter horse and race data, adjust features and parameters, then click **Process Odds** to download a ZIP containing input and adjusted odds.

---

### 2. Running the Script via CLI

You can also run the adjustment directly via command line:

```bash
python adjust_odds.py \
    --input input.csv \
    --output output.csv \
    --w_market 0.6 \
    --alpha 0.97 \
    --shrink 1.0 \
    --round_step 10 \
    --decimals 3
```

* `--input` → CSV file containing horse data.
* `--output` → CSV file to save adjusted odds.
* `--w_market` → weight for blending market probability and feature-adjusted probability.
* `--alpha` → favorite-longshot skew exponent.
* `--shrink` → shrink factor for feature weights.
* `--round_step` → round adjusted odds to nearest step.
* `--decimals` → number of decimals for probability columns.

---

## Input CSV Format

Required columns:

| Column           | Description                            |
| ---------------- | -------------------------------------- |
| RaceID           | Unique race identifier (e.g., R1, R2)  |
| Horse            | Horse name                             |
| BaselineOdds     | Original odds (profit-style)           |
| JockeyScore      | Feature score (0–1)                    |
| HorseWeightScore | Feature score (0–1)                    |
| DrawScore        | Feature score (0–1)                    |
| GoingScore       | Feature score (0–1)                    |
| PaceScore        | Feature score (0–1)                    |
| Overround        | Optional bookmaker margin for the race |

> If a race does not have an Overround value, the script will use the sum of the market-implied probabilities from the baseline odds for that race.

---

## How Odds Are Adjusted

1. Convert **baseline odds** to market probabilities.
2. Normalize probabilities per race.
3. Apply per-horse feature adjustments:

   * Features are centered around 0.5.
   * Feature effect = `(score - 0.5) * beta`
   * Adjustments applied in **log-space** for multiplicative effect.
4. Blend market probabilities and feature-adjusted probabilities (`w_market` weight).
5. Apply favorite-longshot skew (`alpha` exponent).
6. Scale probabilities with the race overround.
7. Convert back to integer profit-style odds.

---

## License

MIT License – feel free to use and modify this project.

---

---

## Live App

Check out the deployed app here: [Horse Racing Odds Adjuster](https://horse-odds-app.onrender.com/)
