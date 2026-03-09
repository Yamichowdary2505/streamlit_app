import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease — Logistic Regression",
    page_icon="🫀",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main-header {
    background: linear-gradient(135deg, #c0392b 0%, #8e1a12 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(192,57,43,0.3);
}
.main-header h1 { color: white; font-size: 2.4rem; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1rem; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.metric-card .label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #e74c3c; font-size: 2rem; font-weight: 600; }

.section-title {
    color: #e74c3c;
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Load & Train ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("heart.csv")
    X  = df.drop("target", axis=1).values
    y  = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)
    Xte     = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(Xtr, y_train)

    return df, X, y, X_train, X_test, y_train, y_test, Xtr, Xte, model, scaler

df, X, y, X_train, X_test, y_train, y_test, Xtr, Xte, model, scaler = load_and_train()

y_train_pred = model.predict(Xtr)
y_test_pred  = model.predict(Xte)
train_acc    = accuracy_score(y_train, y_train_pred)
test_acc     = accuracy_score(y_test,  y_test_pred)
gap          = train_acc - test_acc
train_mse    = mean_squared_error(y_train, y_train_pred)
test_mse     = mean_squared_error(y_test,  y_test_pred)
train_r2     = r2_score(y_train, y_train_pred)
test_r2      = r2_score(y_test,  y_test_pred)
cm           = confusion_matrix(y_test, y_test_pred)

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc   = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
cv_f1    = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted")
cv_prec  = cross_val_score(pipeline, X, y, cv=cv, scoring="precision_weighted")
y_cv     = cross_val_predict(pipeline, X, y, cv=cv)
cm_cv    = confusion_matrix(y, y_cv)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 Heart Disease Prediction</h1>
    <p>Logistic Regression Analysis Dashboard &nbsp;|&nbsp; Heart Disease Dataset &nbsp;|&nbsp; 1025 Samples</p>
</div>
""", unsafe_allow_html=True)

# ─── Top Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="label">Test Accuracy</div><div class="value">{test_acc*100:.1f}%</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="label">CV Accuracy</div><div class="value">{cv_acc.mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="label">F1 Score</div><div class="value">{cv_f1.mean():.3f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="label">Train/Test Gap</div><div class="value">{gap*100:.1f}%</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="label">Total Samples</div><div class="value">1025</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 MSE & R²",
    "🔲 Confusion Matrix",
    "⚖️ Overfitting Check",
    "🔁 Cross Validation",
    "🔮 CV Predict"
])

# ── Tab 1: MSE & R² ───────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">MSE & R² Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mean Squared Error (MSE)**")
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        bars = ax.bar(["Train","Test"], [train_mse, test_mse], color=["#3498db","#e74c3c"], width=0.4, edgecolor="none")
        ax.set_ylabel("MSE", color="#888")
        ax.tick_params(colors="#888")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
        for bar, val in zip(bars, [train_mse, test_mse]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{val:.4f}", ha="center", color="white", fontsize=11)
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown("**R² Score**")
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        bars = ax.bar(["Train","Test"], [train_r2, test_r2], color=["#2ecc71","#f39c12"], width=0.4, edgecolor="none")
        ax.set_ylabel("R²", color="#888")
        ax.tick_params(colors="#888")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
        for bar, val in zip(bars, [train_r2, test_r2]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.4f}", ha="center", color="white", fontsize=11)
        st.pyplot(fig); plt.close()
    st.markdown(f"""
    | Metric | Train | Test |
    |--------|-------|------|
    | MSE    | {train_mse:.4f} | {test_mse:.4f} |
    | R²     | {train_r2:.4f} | {test_r2:.4f}  |
    """)

# ── Tab 2: Confusion Matrix ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["No Disease","Has Disease"],
                    yticklabels=["No Disease","Has Disease"],
                    linewidths=2, linecolor="#0f1117",
                    annot_kws={"size": 16, "weight": "bold"})
        ax.set_xlabel("Predicted", color="#888")
        ax.set_ylabel("Actual",    color="#888")
        ax.tick_params(colors="#ccc")
        ax.set_title("Test Set Confusion Matrix", color="white", pad=12)
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown(f"""
        **Test Set Results:**
        | | Predicted No Disease | Predicted Has Disease |
        |---|---|---|
        | **Actual No Disease**  | ✅ {cm[0][0]} | ❌ {cm[0][1]} |
        | **Actual Has Disease** | ❌ {cm[1][0]} | ✅ {cm[1][1]} |

        **Accuracy : {test_acc*100:.2f}%**
        """)
        st.text(classification_report(y_test, y_test_pred, target_names=["No Disease","Has Disease"]))

# ── Tab 3: Overfitting Check ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Overfitting / Underfitting Check</div>', unsafe_allow_html=True)
    if gap > 0.10:
        status = "❌ OVERFITTING DETECTED"
    elif test_acc < 0.75:
        status = "❌ UNDERFITTING DETECTED"
    else:
        status = "✅ MODEL IS HEALTHY"

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        bars = ax.bar(["Train Accuracy","Test Accuracy"],
                      [train_acc*100, test_acc*100],
                      color=["#3498db","#e74c3c"], width=0.4, edgecolor="none")
        ax.set_ylim(60, 100)
        ax.set_ylabel("Accuracy %", color="#888")
        ax.tick_params(colors="#888")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
        for bar, val in zip(bars, [train_acc*100, test_acc*100]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val:.2f}%", ha="center", color="white", fontsize=11)
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Train Accuracy | {train_acc*100:.2f}% |
        | Test  Accuracy | {test_acc*100:.2f}%  |
        | Gap            | {gap*100:.2f}%        |
        | Status         | {status}              |

        **Reference Guide:**
        - Gap < 5%   → No overfitting ✅
        - Gap 5-10%  → Slight overfitting ⚠️
        - Gap > 10%  → Overfitting ❌
        - Test < 75% → Underfitting ❌
        """)

# ── Tab 4: Cross Validation ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">5-Fold Cross Validation Results</div>', unsafe_allow_html=True)
    cv_df = pd.DataFrame({
        "Fold"     : [f"Fold {i+1}" for i in range(5)],
        "Accuracy" : cv_acc,
        "F1 Score" : cv_f1,
        "Precision": cv_prec
    })
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        x = np.arange(5)
        ax.plot(x, cv_acc*100,  marker="o", color="#e74c3c", label="Accuracy",  linewidth=2)
        ax.plot(x, cv_f1*100,   marker="s", color="#3498db", label="F1 Score",  linewidth=2)
        ax.plot(x, cv_prec*100, marker="^", color="#2ecc71", label="Precision", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i+1}" for i in range(5)], color="#888")
        ax.set_ylabel("Score %", color="#888")
        ax.tick_params(colors="#888")
        ax.legend(facecolor="#1a1d27", labelcolor="white", edgecolor="#2a2d3a")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2d3a")
        ax.set_ylim(75, 100)
        st.pyplot(fig); plt.close()
    with col2:
        st.dataframe(cv_df.style.format({"Accuracy":"{:.4f}","F1 Score":"{:.4f}","Precision":"{:.4f}"}), use_container_width=True)
        st.markdown(f"""
        | Metric | Mean | Std |
        |--------|------|-----|
        | Accuracy  | {cv_acc.mean()*100:.2f}%  | ±{cv_acc.std()*100:.2f}%  |
        | F1 Score  | {cv_f1.mean():.4f}   | ±{cv_f1.std():.4f}   |
        | Precision | {cv_prec.mean():.4f} | ±{cv_prec.std():.4f} |
        """)

# ── Tab 5: Cross Val Predict ──────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Cross Validate Predict Results</div>', unsafe_allow_html=True)
    cv_accuracy = accuracy_score(y, y_cv)
    cv_mse      = mean_squared_error(y, y_cv)
    cv_r2       = r2_score(y, y_cv)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        sns.heatmap(cm_cv, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["No Disease","Has Disease"],
                    yticklabels=["No Disease","Has Disease"],
                    linewidths=2, linecolor="#0f1117",
                    annot_kws={"size":16,"weight":"bold"})
        ax.set_xlabel("Predicted", color="#888")
        ax.set_ylabel("Actual",    color="#888")
        ax.tick_params(colors="#ccc")
        ax.set_title("CV Predict Confusion Matrix", color="white", pad=12)
        st.pyplot(fig); plt.close()
    with col2:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Overall Accuracy | {cv_accuracy*100:.2f}% |
        | MSE              | {cv_mse:.4f}            |
        | R² Score         | {cv_r2:.4f}             |
        | Total Samples    | {len(y)}                |
        | Folds            | 5                       |
        """)
        st.text(classification_report(y, y_cv, target_names=["No Disease","Has Disease"]))

    st.markdown("**First 20 Sample Predictions:**")
    labels    = ["No Disease", "Has Disease"]
    sample_df = pd.DataFrame({
        "Sample"   : range(1, 21),
        "Actual"   : [labels[y[i]]    for i in range(20)],
        "Predicted": [labels[y_cv[i]] for i in range(20)],
        "Correct"  : ["✅" if y[i] == y_cv[i] else "❌" for i in range(20)]
    })
    st.dataframe(sample_df, use_container_width=True)
