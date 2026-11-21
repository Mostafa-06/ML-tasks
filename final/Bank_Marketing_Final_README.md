# Bank Marketing: Term Deposit Prediction

Binary classification project that predicts whether a bank client will **subscribe to a term deposit** (`deposit` = yes / no) from client details and data about the marketing campaign. The notebook covers data exploration, outlier analysis, preprocessing, a comparison of 14 classifiers, 5-fold cross-validation, feature importance, and a Gradio demo app.

## Dataset

- **Size:** 11,162 rows × 17 columns (16 after dropping `poutcome`), with no missing values and no duplicates
- **Target:** `deposit` (yes / no), with fairly balanced classes (about 47% yes in the training set)
- **Features:**

| Group | Columns |
|---|---|
| Client | `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan` |
| Last contact | `contact`, `day`, `month`, `duration` |
| Campaign history | `campaign`, `pdays`, `previous` (`poutcome` is dropped) |

> The CSV is not included in this repo. Place it next to the notebook as `bank.csv`.

## Workflow

1. **Cleaning**: check missing values and duplicates, drop `poutcome`
2. **EDA**: count plots, box plots, histograms (with KDE), correlation heatmap, and an IQR outlier count per numeric column
3. **Train/test split**: 80/20, `random_state=42` (8,929 train / 2,233 test)
4. **Preprocessing**: label encoding for categorical features and the target, standard scaling for numeric features
5. **Modeling**: 13 individual models and 1 stacking classifier
6. **Cross-validation**: 5-fold CV on LightGBM, CatBoost, XGBoost, and Random Forest
7. **Evaluation**: confusion matrix, classification report, ROC curve, and precision-recall curve for every model
8. **Feature importance**: from CatBoost
9. **Demo**: a Gradio app that predicts for a single client using the stacking model

## Models Compared

| Family | Models |
|---|---|
| Linear | Logistic Regression |
| Distance-based | KNN |
| Kernel | SVC |
| Tree-based | Decision Tree |
| Probabilistic | Gaussian Naive Bayes |
| Bagging | Random Forest, Extra Trees |
| Boosting | AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, LightGBM |
| Ensembles | Stacking (LR + RF + NB + LightGBM → XGBoost) |

## Results

Test-set performance for the `yes` class, ranked by F1:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **CatBoost** | 0.850 | 0.826 | 0.870 | 0.847 | 0.923 |
| XGBoost | 0.844 | 0.822 | 0.858 | 0.840 | 0.916 |
| Hist Gradient Boosting | 0.842 | 0.815 | 0.865 | 0.839 | 0.919 |
| LightGBM | 0.841 | 0.812 | 0.867 | 0.839 | 0.919 |
| Random Forest | 0.837 | 0.807 | 0.866 | 0.835 | 0.908 |
| Stacking Classifier | 0.833 | 0.796 | 0.876 | 0.834 | 0.907 |
| Extra Trees | 0.828 | 0.804 | 0.848 | 0.825 | 0.905 |
| Gradient Boosting | 0.825 | 0.801 | 0.844 | 0.822 | 0.909 |
| SVC | 0.808 | 0.785 | 0.826 | 0.805 | 0.883 |
| AdaBoost | 0.791 | 0.786 | 0.773 | 0.780 | 0.875 |
| Logistic Regression | 0.786 | 0.788 | 0.754 | 0.771 | 0.865 |
| Decision Tree | 0.767 | 0.755 | 0.758 | 0.756 | 0.766 |
| Naive Bayes | 0.743 | 0.696 | 0.820 | 0.753 | 0.821 |
| KNN | 0.766 | 0.777 | 0.717 | 0.746 | 0.836 |

**5-fold cross-validation** (on the training set) agrees with the test results:

| Model | CV Accuracy | CV F1 |
|---|---|---|
| CatBoost | 0.863 | 0.859 |
| LightGBM | 0.858 | 0.854 |
| XGBoost | 0.855 | 0.851 |
| Random Forest | 0.844 | 0.840 |

**Top features (CatBoost importance):** `duration` (26%), `month` (20%), `day` (10%), `contact` (8%), `age` (6%).

**Takeaways**

- CatBoost is the best model, with about 85% accuracy and an ROC-AUC of 0.92.
- Random Forest, Extra Trees, and Decision Tree reach 100% train accuracy, which indicates overfitting.
- The stacking ensemble did not beat the best single boosting models.
- Call `duration` is the most important feature. Since it is only known after the call ends, a model meant for real use before calling should be tested without it.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Gradio · Matplotlib · Seaborn

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost gradio tqdm jupyter

# 3. Add the dataset as "bank.csv" in the project folder
# 4. Update the path in the first data-loading cell, then run:
jupyter notebook final.ipynb
```

The notebook was written in Google Colab. If you run it there, upload the CSV to `/content/`.

## Project Structure

```
├── final.ipynb   # Full analysis, modeling, and demo app
├── bank.csv      # Dataset (add separately)
└── README.md
```

## Possible Improvements

- Test the models without `duration` for a realistic pre-call prediction
- Hyperparameter tuning (GridSearchCV / Optuna) to reduce overfitting
- One-hot encoding for nominal features such as `job` and `month`
- SHAP analysis for individual predictions
- Deploy the Gradio demo (for example on Hugging Face Spaces)

## Author

**Mostafa**, Computer Science student (Scientific Computing)
