import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_curve, auc
)
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facebook Ads Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7c6af7; }
    .metric-label { font-size: 0.85rem; color: #8b92b0; margin-top: 4px; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #e2e8f0;
        border-left: 4px solid #7c6af7; padding-left: 12px;
        margin: 28px 0 16px 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px; color: #8b92b0;
        border: 1px solid #2e3250; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #7c6af7 !important; color: white !important; border-color: #7c6af7 !important;
    }
    .badge-clicked { background:#22c55e22; color:#22c55e; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
    .badge-not { background:#ef444422; color:#ef4444; padding:3px 10px; border-radius:20px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
DEFAULT_CSV = Path(__file__).parent / "facebook_ads.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path, encoding='latin1')

@st.cache_resource
def train_model(df):
    features = df.drop(['Names', 'emails', 'Country', 'Clicked'], axis=1)
    target = df['Clicked']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.20, random_state=42, shuffle=True
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, scaler, X_test, y_test, y_pred, y_prob, features.columns.tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Facebook Ads ML")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Filters")

    if uploaded:
        df_raw = load_data(uploaded)
    else:
        df_raw = load_data(DEFAULT_CSV) if DEFAULT_CSV.exists() else st.error("📂 Please upload your facebook_ads.csv using the uploader above.") or st.stop()

    salary_range = st.slider(
        "Salary Range ($)",
        int(df_raw["Salary"].min()), int(df_raw["Salary"].max()),
        (int(df_raw["Salary"].min()), int(df_raw["Salary"].max()))
    )
    time_range = st.slider(
        "Time Spent on Site (min)",
        float(df_raw["Time Spent on Site"].min()), float(df_raw["Time Spent on Site"].max()),
        (float(df_raw["Time Spent on Site"].min()), float(df_raw["Time Spent on Site"].max()))
    )
    click_filter = st.multiselect("Clicked", [0, 1], default=[0, 1],
                                   format_func=lambda x: "Clicked" if x == 1 else "Not Clicked")

    df = df_raw[
        (df_raw["Salary"].between(*salary_range)) &
        (df_raw["Time Spent on Site"].between(*time_range)) &
        (df_raw["Clicked"].isin(click_filter if click_filter else [0, 1]))
    ]
    st.markdown(f"**{len(df):,}** / {len(df_raw):,} records shown")

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Facebook Ads — EDA & Click Prediction")
st.markdown("Explore ad engagement patterns and predict which users will click.")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
click_rate = df["Clicked"].mean() * 100
avg_salary = df["Salary"].mean()
avg_time = df["Time Spent on Site"].mean()
countries = df["Country"].nunique()

for col, val, label in [
    (c1, f"{len(df):,}", "Total Users"),
    (c2, f"{click_rate:.1f}%", "Click Rate"),
    (c3, f"${avg_salary:,.0f}", "Avg Salary"),
    (c4, f"{avg_time:.1f} min", "Avg Time on Site"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Deep Dive", "🌍 Geography", "🤖 ML Model"])

COLORS = {"clicked": "#7c6af7", "not_clicked": "#f97316", "purple": "#7c6af7", "orange": "#f97316"}
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8cdd8", family="Inter"),
    margin=dict(t=40, b=20, l=10, r=10),
    xaxis=dict(gridcolor="#1e2130", showline=False),
    yaxis=dict(gridcolor="#1e2130", showline=False),
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Click Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        counts = df["Clicked"].value_counts().reset_index()
        counts.columns = ["Clicked", "Count"]
        counts["Label"] = counts["Clicked"].map({1: "Clicked", 0: "Not Clicked"})
        fig = px.pie(counts, values="Count", names="Label",
                     color="Label",
                     color_discrete_map={"Clicked": COLORS["clicked"], "Not Clicked": COLORS["not_clicked"]},
                     hole=0.55)
        fig.update_layout(**PLOT_LAYOUT, title="Click vs No-Click", showlegend=True,
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_traces(textinfo="percent+label", textfont_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.histogram(df, x="Salary", color=df["Clicked"].map({1:"Clicked",0:"Not Clicked"}),
                            barmode="overlay", nbins=30,
                            color_discrete_map={"Clicked": COLORS["clicked"], "Not Clicked": COLORS["not_clicked"]},
                            labels={"color": "Status", "Salary": "Salary ($)"},
                            title="Salary Distribution by Click Status")
        fig2.update_layout(**PLOT_LAYOUT, legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig2.update_traces(opacity=0.8)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Time on Site Distribution</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        fig3 = px.histogram(df, x="Time Spent on Site",
                            color=df["Clicked"].map({1:"Clicked",0:"Not Clicked"}),
                            barmode="overlay", nbins=30,
                            color_discrete_map={"Clicked": COLORS["clicked"], "Not Clicked": COLORS["not_clicked"]},
                            title="Time on Site Distribution")
        fig3.update_layout(**PLOT_LAYOUT, legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig3.update_traces(opacity=0.8)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        fig4 = px.box(df, x=df["Clicked"].map({1:"Clicked",0:"Not Clicked"}),
                      y="Time Spent on Site",
                      color=df["Clicked"].map({1:"Clicked",0:"Not Clicked"}),
                      color_discrete_map={"Clicked": COLORS["clicked"], "Not Clicked": COLORS["not_clicked"]},
                      title="Time on Site — Box Plot")
        fig4.update_layout(**PLOT_LAYOUT, showlegend=False,
                           xaxis_title="Status", yaxis_title="Time Spent on Site (min)")
        st.plotly_chart(fig4, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Deep Dive
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Salary vs Time Spent on Site</div>', unsafe_allow_html=True)

    fig5 = px.scatter(df, x="Time Spent on Site", y="Salary",
                      color=df["Clicked"].map({1:"Clicked",0:"Not Clicked"}),
                      color_discrete_map={"Clicked": COLORS["clicked"], "Not Clicked": COLORS["not_clicked"]},
                      opacity=0.7, title="Salary vs Time Spent — Click Segmentation",
                      labels={"Time Spent on Site": "Time on Site (min)", "Salary": "Salary ($)"})
    fig5.update_layout(**PLOT_LAYOUT, legend=dict(bgcolor="rgba(0,0,0,0)"), height=420)
    fig5.update_traces(marker=dict(size=7))
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df = df[["Time Spent on Site", "Salary", "Clicked"]]
    corr = num_df.corr()
    fig6 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#f97316"], [0.5, "#1e2130"], [1, "#7c6af7"]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True, zmin=-1, zmax=1
    ))
    fig6.update_layout(**PLOT_LAYOUT, title="Feature Correlation Matrix", height=350)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-header">Salary Segments & Click Rate</div>', unsafe_allow_html=True)
    df2 = df.copy()
    df2["Salary Bucket"] = pd.cut(df2["Salary"], bins=5,
                                   labels=["Very Low", "Low", "Medium", "High", "Very High"])
    bucket_click = df2.groupby("Salary Bucket")["Clicked"].mean().reset_index()
    bucket_click.columns = ["Salary Bucket", "Click Rate"]
    bucket_click["Click Rate %"] = (bucket_click["Click Rate"] * 100).round(1)

    fig7 = px.bar(bucket_click, x="Salary Bucket", y="Click Rate %",
                  color="Click Rate %",
                  color_continuous_scale=["#f97316", "#7c6af7"],
                  title="Click Rate by Salary Segment",
                  text="Click Rate %")
    fig7.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="Click Rate (%)")
    fig7.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig7, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Geography
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Country-level Stats</div>', unsafe_allow_html=True)
    geo = df.groupby("Country").agg(
        Users=("Clicked", "count"),
        Click_Rate=("Clicked", "mean"),
        Avg_Salary=("Salary", "mean"),
        Avg_Time=("Time Spent on Site", "mean"),
    ).reset_index()
    geo["Click Rate %"] = (geo["Click_Rate"] * 100).round(1)

    fig8 = px.choropleth(geo, locations="Country", locationmode="country names",
                          color="Click Rate %",
                          color_continuous_scale=["#1e2130", "#7c6af7"],
                          title="Click Rate by Country (%)",
                          hover_data=["Users", "Avg_Salary", "Avg_Time"])
    fig8.update_layout(**PLOT_LAYOUT, geo=dict(bgcolor="rgba(0,0,0,0)",
                                                lakecolor="rgba(0,0,0,0)",
                                                showocean=True, oceancolor="#0d1117",
                                                showland=True, landcolor="#1a1d2e",
                                                showcountries=True, countrycolor="#2e3250"),
                       height=460)
    st.plotly_chart(fig8, use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        top10_users = geo.nlargest(10, "Users")[["Country", "Users", "Click Rate %"]]
        fig9 = px.bar(top10_users, x="Users", y="Country", orientation="h",
                      color="Click Rate %", color_continuous_scale=["#f97316", "#7c6af7"],
                      title="Top 10 Countries by User Count")
        fig9.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        fig9.update_yaxes(autorange="reversed", gridcolor="#1e2130")
        st.plotly_chart(fig9, use_container_width=True)

    with col_f:
        top10_cr = geo[geo["Users"] >= 3].nlargest(10, "Click Rate %")[["Country", "Click Rate %", "Users"]]
        fig10 = px.bar(top10_cr, x="Click Rate %", y="Country", orientation="h",
                       color="Click Rate %", color_continuous_scale=["#f97316", "#7c6af7"],
                       title="Top 10 Countries by Click Rate (min 3 users)")
        fig10.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        fig10.update_yaxes(autorange="reversed", gridcolor="#1e2130")
        st.plotly_chart(fig10, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — ML Model
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Logistic Regression — Click Predictor</div>', unsafe_allow_html=True)

    model, scaler, X_test, y_test, y_pred, y_prob, feature_names = train_model(df_raw)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, f"{acc*100:.1f}%", "Accuracy"),
        (m2, f"{f1:.3f}", "F1 Score"),
        (m3, f"{roc_auc:.3f}", "AUC-ROC"),
        (m4, f"{len(X_test)}", "Test Samples"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_g, col_h = st.columns(2)

    with col_g:
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: Not Clicked", "Pred: Clicked"],
            y=["Actual: Not Clicked", "Actual: Clicked"],
            colorscale=[[0, "#1a1d2e"], [1, "#7c6af7"]],
            text=cm, texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(**PLOT_LAYOUT, title="Confusion Matrix", height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_h:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color="#8b92b0"), name="Random"))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      line=dict(color="#7c6af7", width=2.5),
                                      name=f"AUC = {roc_auc:.3f}",
                                      fill="tozeroy", fillcolor="rgba(124,106,247,0.15)"))
        fig_roc.update_layout(**PLOT_LAYOUT, title="ROC Curve",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate",
                               legend=dict(bgcolor="rgba(0,0,0,0)"), height=340)
        st.plotly_chart(fig_roc, use_container_width=True)

    # Classification Report
    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    report_df = report_df.drop(index=["accuracy"], errors="ignore")
    report_df.index = report_df.index.map({"0": "Not Clicked", "1": "Clicked",
                                            "macro avg": "Macro Avg",
                                            "weighted avg": "Weighted Avg"})
    st.dataframe(report_df.style.background_gradient(cmap="Purples", axis=0), use_container_width=True)

    # Feature importance (coefficients)
    st.markdown('<div class="section-header">Feature Importance (Coefficients)</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient")
    fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                      color="Coefficient",
                      color_continuous_scale=["#f97316", "#7c6af7"],
                      title="Logistic Regression Coefficients")
    fig_coef.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=260)
    st.plotly_chart(fig_coef, use_container_width=True)

    # Live prediction
    st.markdown('<div class="section-header">🔮 Live Prediction</div>', unsafe_allow_html=True)
    with st.form("predict_form"):
        p1, p2 = st.columns(2)
        with p1:
            inp_time = st.slider("Time Spent on Site (min)", 5.0, 60.0, 30.0, 0.5)
        with p2:
            inp_salary = st.slider("Salary ($)", 20.0, 100000.0, 52000.0, 500.0)
        submitted = st.form_submit_button("Predict Click Probability", use_container_width=True)

    if submitted:
        inp_arr = np.array([[inp_time, inp_salary]])
        inp_scaled = scaler.transform(inp_arr)
        prob = model.predict_proba(inp_scaled)[0][1]
        pred = model.predict(inp_scaled)[0]
        badge = "badge-clicked" if pred == 1 else "badge-not"
        label = "✅ Likely to Click" if pred == 1 else "❌ Unlikely to Click"

        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{prob*100:.1f}%</div>
                <div class="metric-label">Click Probability</div>
            </div>""", unsafe_allow_html=True)
        with res2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value"><span class="{badge}">{label}</span></div>
                <div class="metric-label">Prediction</div>
            </div>""", unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "#7c6af7", "size": 36}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#8b92b0"),
                bar=dict(color="#7c6af7"),
                bgcolor="#1a1d2e",
                steps=[
                    dict(range=[0, 40], color="#ef444422"),
                    dict(range=[40, 60], color="#f9731622"),
                    dict(range=[60, 100], color="#22c55e22"),
                ],
                threshold=dict(line=dict(color="#7c6af7", width=3), value=50),
            ),
            title={"text": "Click Probability", "font": {"color": "#c8cdd8"}},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8cdd8"),
                                 height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#4a5068;font-size:0.8rem;'>Facebook Ads Analytics · Logistic Regression Click Predictor</div>",
    unsafe_allow_html=True
)