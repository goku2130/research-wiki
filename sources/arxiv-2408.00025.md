---
id: arxiv:2408.00025
type: paper
title: 'Need of AI in Modern Education: in the Eyes of Explainable AI (xAI)'
url: https://arxiv.org/abs/2408.00025
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

# Research Summary: Need of AI in Modern Education: in the Eyes of Explainable AI (xAI)

### Core Problem
While Artificial Intelligence (AI) offers significant potential for personalized learning and administrative efficiency in modern education, its "black-box" nature creates an "explanation gap." This lack of transparency is particularly critical in high-stakes educational contexts where socio-economic factors, such as parental income, heavily influence educational access. The authors investigate whether AI models used to predict income (as a proxy for the ability to afford modern education) harbor intrinsic biases and whether standard Explainable AI (xAI) tools are sufficient to uncover these biases.

### Method/Recipe
The researchers employed a multi-stage pipeline to train a predictive model and analyze its decision-making process:

1.  **Data and Model Training**: The study used the *Adult Census Income* dataset from the UCI Machine Learning Repository to perform binary classification, predicting whether an individual earns more than \$50,000 USD per year. An **XGBoost** model was trained for this task.
2.  **Feature Importance Analysis**:
    *   **Permutation Importance (PI)**: Used the ELI5 library to quantify importance based on the drop in AUC when a feature is removed.
    *   **XGBoost Gain**: Evaluated the relative contribution of features across all trees.
    *   **SHAP (Shapley Additive Explanations)**: Calculated the mean absolute SHAP values for global importance and used Summary and Dependency plots to visualize feature interactions.
3.  **Local and Global Explanations**:
    *   **Local**: Applied **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (Force and Decision plots) to explain individual predictions.
    *   **Global Surrogates**: Trained a Decision Tree and Logistic Regression model to mimic the XGBoost model's predictions.
    *   **Representative Sampling**: Used the **SP-LIME** (Submodular Picking) algorithm to select a diverse set of representative instances for global understanding.
4.  **Fairness Audit**: Utilized the **FairML** library to perform orthogonal projections of input features to quantify the model's dependency on sensitive attributes (e.g., race, sex, nationality).

### Key Formulas
The study utilizes the Shapley value to allocate "payout" (prediction) among features fairly:

$$
\phi_{i}=\sum_{S\subseteq F\setminus\{i\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}\left[f_{S\cup\{i\}}\left(x_{S\cup\{i\}}\right)-f_{S}\left(x_{S}\right)\right]
$$

To quantify bias, FairML measures the change in model output $\Delta F$ resulting from a modified input $x'$:

$$
\Delta F\big(x^{\prime},y\big)=F\big(x^{\prime},y\big)-F\big(x,y\big)
$$

### Key Quantitative Results
**XGBoost Model Performance (Test Set):**
*   **Accuracy**: 0.87
*   **Precision**: 0.78
*   **Recall**: 0.62
*   **F1 Score**: 0.69
*   **AUC**: 0.92

**Global Surrogate Model Performance (Test Set):**
*   **Decision Tree**: Accuracy 0.85, AUC 0.865, and $R^2$ of 0.79.
*   **Logistic Regression**: $R^2$ of -1.42 (indicating a poor fit compared to the mean).

**Feature Importance**: Across PI, XGBoost Gain, and SHAP, the most significant features were consistently `married_1`, `education.num`, and `capital.gain`.

### Stated Limitations and Findings
*   **LIME Instability**: The authors found LIME to be "inherently unstable," frequently producing varied explanations for identical instances due to randomness in surrogate model generation.
*   **Detection Gap**: Standard feature importance methods (like SHAP and XGBoost Gain) failed to highlight certain intrinsic biases.
*   **Algorithmic Bias**: FairML revealed strong biases that other xAI tools missed; the model predicted a higher likelihood of earning $>\$50\text{k}$ if the individual was **White**, **Male**, and a **United States national**.
*   **Conclusion on Fairness**: The authors conclude that "All is simply, not Fair," noting that biases in training data are often perpetuated or amplified by AI, which could restrict educational opportunities for disadvantaged groups.
