# CarbonScope — Carbon Emission Prediction Using Machine Learning

A Flask-based educational ML project for predicting carbon emissions and demonstrating the ML lifecycle.

## Project structure

```text
CarbonScope_ML_Project/
├── app.py
├── train.py
├── predict.py
├── analyze.py
├── evaluate.py
├── unsupervised.py
├── requirements.txt
├── README.md
├── dataset/
│   └── carbon_emissions.csv
├── models/
│   ├── linear_regression.joblib
│   ├── ridge.joblib
│   ├── lasso.joblib
│   ├── elastic_net.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── metadata.json
├── processed_data/
│   ├── final_dataset.csv
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   ├── evaluation_results.json
│   └── unsupervised_results.json
├── templates/index.html
└── static/
    ├── style.css
    └── script.js
```

## Run in VS Code

```powershell
cd CarbonScope_ML_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in Chrome.

To retrain all artifacts manually:

```powershell
python train.py
python analyze.py
python evaluate.py
python unsupervised.py
```

The interface is intentionally structured like the supplied SmartCrop reference project: a flat root-level Flask project, dataset/models/processed_data folders, one template, one stylesheet and one JavaScript file. The visual design is changed to an environmental green/teal theme with rounded cards and a cleaner dashboard layout.
