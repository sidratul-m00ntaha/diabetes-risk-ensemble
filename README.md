<<<<<<< HEAD
# Diabetes Risk Classification Using Machine Learning

A comparative study of 9 supervised learning algorithms + 2 ensemble 
strategies for three-class diabetes risk prediction (Low Risk / 
Prediabetes / High Risk).

## Results

| Model | Accuracy | F1 (Weighted) | AUC-ROC |
|---|---|---|---|
| **Voting Ensemble (top 3) ★** | 0.9150 | 0.9147 | **0.9877** |
| Logistic Regression | 0.9183 | 0.9186 | 0.9873 |
| Neural Network (MLP) | 0.9200 | 0.9200 | 0.9873 |
| SVM (RBF) | 0.9125 | 0.9120 | 0.9863 |
| Gradient Boosting | 0.9058 | 0.9067 | 0.9850 |
| Random Forest | 0.9075 | 0.9069 | 0.9846 |

## Dataset
6,000 patient records × 17 clinical and lifestyle features.  
Source: [https://www.kaggle.com/datasets/vishardmehta/diabetes-risk-prediction-dataset]

## How to Run

```bash
git clone https://github.com/sidratul-m00ntaha/diabetes-risk-ensemble
cd diabetes-risk-ensemble
pip install -r requirements.txt
python src/pipeline.py
```

## Project Structure
See the diagram in `outputs/system_architecture.svg`
=======
# diabetes-risk-ensemble
Domain-driven ensemble ML framework for multi-class diabetes risk prediction and decision support
>>>>>>> 74429331c44006e6cac55aabbe90b1a4cd7204f0
