import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Import the custom transformer class so joblib can deserialize the model correctly
from train_model import TitanicFeatureEngineer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4338ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.05rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }

    /* ── Style st.container(border=True) as glassmorphic cards ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #0b0f19 !important; }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background-color: rgba(255,255,255,0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }

    /* ── Buttons ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        box-shadow: 0 6px 20px rgba(99,102,241,0.6);
        transform: translateY(-1px);
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
# @st.cache_resource
def load_model_artifacts():
    if not os.path.exists('titanic_model.joblib') or not os.path.exists('titanic_metrics.joblib'):
        return None, None
    return joblib.load('titanic_model.joblib'), joblib.load('titanic_metrics.joblib')

@st.cache_data
def load_raw_dataset():
    if os.path.exists('Titanic-Dataset.csv'):
        return pd.read_csv('Titanic-Dataset.csv')
    return None

pipeline, metrics = load_model_artifacts()
raw_df = load_raw_dataset()

if pipeline is None or metrics is None:
    st.error("🚨 Model not found! Run `python train_model.py` first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🚢 TITANIC SURVIVAL PREDICTOR</h1>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Survival Predictor",
    "📊 Exploratory Data Analysis",
    "⚙️ Model Insights & Metrics",
    "📁 Batch Predictions"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — SURVIVAL PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.write("")
    col_input, col_result = st.columns([1, 1], gap="large")

    # ── Left: inputs ──────────────────────────────────────────
    with col_input:
        st.markdown("#### 🧾 Passenger Configuration")
        st.caption("Adjust the passenger details to compute a live survival prediction.")

        name_input = st.text_input("Passenger Name", value="Braund, Mr. Owen Harris")

        # Title extraction
        extracted_title = "Mr"
        if "," in name_input and "." in name_input:
            try:
                raw_title = name_input.split(',')[1].split('.')[0].strip()
                title_map = {
                    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
                    'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Mrs',
                    'Countess': 'Mrs', 'Dona': 'Mrs', 'Dr': 'Officer', 'Rev': 'Officer',
                    'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
                    'Sir': 'Noble', 'Don': 'Noble', 'Jonkheer': 'Noble'
                }
                extracted_title = title_map.get(raw_title, 'Mr')
            except Exception:
                extracted_title = "Mr"

        col_g, col_p = st.columns(2)
        with col_g:
            sex = st.radio("Gender", ["Male", "Female"], horizontal=True)
        with col_p:
            pclass = st.selectbox(
                "Ticket Class",
                [1, 2, 3],
                index=2,
                format_func=lambda x: f"{x}st Class" if x == 1 else (f"{x}nd Class" if x == 2 else f"{x}rd Class")
            )

        col_a, col_f = st.columns(2)
        with col_a:
            age = st.slider("Age", min_value=0.42, max_value=80.0, value=28.0, step=1.0)
        with col_f:
            fare = st.number_input(
                "Fare Paid (£)",
                min_value=0.0, max_value=512.33,
                value={1: 84.0, 2: 20.6, 3: 13.6}[pclass],
                step=1.0
            )

        col_sib, col_par = st.columns(2)
        with col_sib:
            sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=8, value=0)
        with col_par:
            parch = st.number_input("Parents / Children", min_value=0, max_value=6, value=0)

        embarked = st.selectbox(
            "Port of Embarkation",
            ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"]
        )
        embarked_code = embarked[-2]

    # ── Right: prediction result ──────────────────────────────
    with col_result:
        st.markdown("#### 📊 Survival Probability")
        st.caption("Real-time AI prediction based on your passenger configuration.")

        input_data = pd.DataFrame([{
            'PassengerId': 999,
            'Pclass': pclass,
            'Name': name_input,
            'Sex': sex.lower(),
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Ticket': '12345',
            'Fare': fare,
            'Cabin': np.nan,
            'Embarked': embarked_code
        }])

        try:
            proba = pipeline.predict_proba(input_data)[0][1]
            prediction = pipeline.predict(input_data)[0]
        except Exception as err:
            st.error(f"Prediction error: {err}")
            st.stop()

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 48, 'family': 'Outfit', 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "#475569"},
                'bar': {'color': "#6366f1"},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 1.5,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 35],  'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [35, 65], 'color': 'rgba(245,158,11,0.15)'},
                    {'range': [65, 100],'color': 'rgba(16,185,129,0.15)'}
                ],
                'threshold': {
                    'line': {'color': "#818cf8", 'width': 3},
                    'thickness': 0.8,
                    'value': proba * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#94a3b8", 'family': "Plus Jakarta Sans"},
            height=280,
            margin=dict(l=30, r=30, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Outcome card
        if prediction == 1:
            st.success(f"✅ **PREDICTED SURVIVAL** — {proba*100:.1f}% chance of surviving")
        else:
            st.error(f"❌ **PREDICTED DECEASED** — {(1-proba)*100:.1f}% chance of perishing")

        with st.container(border=True):
            st.markdown(f"""
**Extracted Title:** `{extracted_title}` &nbsp;|&nbsp; **Family Size:** `{sibsp + parch + 1}`

**Key factors driving this prediction:**
- 🎟️ **Ticket Class:** 1st Class had ~63% survival vs 3rd Class ~24%
- 👥 **Gender:** Women were ~4× more likely to survive (maritime code)
- 👨‍👩‍👧 **Family Size:** Small families (2–4) had the best survival coordination
""")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.write("")
    if raw_df is None:
        st.warning("Dataset `Titanic-Dataset.csv` not found.")
    else:
        st.markdown("#### 📈 Dataset Overview")

        total_passengers = len(raw_df)
        total_survived   = int(raw_df['Survived'].sum())
        survival_pct     = total_survived / total_passengers * 100

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            with st.container(border=True):
                st.markdown(f'<div class="metric-value">{total_passengers}</div><div class="metric-label">Total Passengers</div>', unsafe_allow_html=True)
        with col_m2:
            with st.container(border=True):
                st.markdown(f'<div class="metric-value" style="color:#10b981">{total_survived}</div><div class="metric-label">Survivors</div>', unsafe_allow_html=True)
        with col_m3:
            with st.container(border=True):
                st.markdown(f'<div class="metric-value" style="color:#f43f5e">{total_passengers - total_survived}</div><div class="metric-label">Perished</div>', unsafe_allow_html=True)
        with col_m4:
            with st.container(border=True):
                st.markdown(f'<div class="metric-value" style="color:#a5b4fc">{survival_pct:.1f}%</div><div class="metric-label">Survival Rate</div>', unsafe_allow_html=True)

        st.write("")

        # Row 1: Gender & Class
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            gender_surv = raw_df.groupby('Sex')['Survived'].mean().reset_index()
            gender_surv['Survival Rate (%)'] = gender_surv['Survived'] * 100
            fig_sex = px.bar(
                gender_surv, x='Sex', y='Survival Rate (%)',
                color='Sex',
                color_discrete_map={'female': '#ec4899', 'male': '#3b82f6'},
                title="Survival Rate by Gender",
                template='plotly_dark'
            )
            fig_sex.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_sex.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                yaxis=dict(range=[0,110], gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=50, b=20, l=10, r=10)
            )
            with st.container(border=True):
                st.plotly_chart(fig_sex, use_container_width=True)

        with col_g2:
            class_surv = raw_df.groupby('Pclass')['Survived'].mean().reset_index()
            class_surv['Survival Rate (%)'] = class_surv['Survived'] * 100
            class_surv['Class'] = class_surv['Pclass'].map({1:'1st Class',2:'2nd Class',3:'3rd Class'})
            fig_cls = px.bar(
                class_surv, x='Class', y='Survival Rate (%)',
                color='Class',
                color_discrete_sequence=['#10b981','#f59e0b','#ef4444'],
                title="Survival Rate by Ticket Class",
                template='plotly_dark'
            )
            fig_cls.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_cls.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                yaxis=dict(range=[0,110], gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=50, b=20, l=10, r=10)
            )
            with st.container(border=True):
                st.plotly_chart(fig_cls, use_container_width=True)

        # Row 2: Age density & Fare box
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            df_clean = raw_df.dropna(subset=['Age']).copy()
            df_clean['Status'] = df_clean['Survived'].map({0:'Perished', 1:'Survived'})
            fig_age = px.histogram(
                df_clean, x='Age', color='Status',
                marginal='box', nbins=40, opacity=0.65,
                color_discrete_map={'Perished':'#f43f5e','Survived':'#10b981'},
                barmode='overlay',
                title="Age Distribution by Survival",
                template='plotly_dark'
            )
            fig_age.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99, bgcolor='rgba(0,0,0,0.4)'),
                margin=dict(t=50, b=20, l=10, r=10)
            )
            with st.container(border=True):
                st.plotly_chart(fig_age, use_container_width=True)

        with col_g4:
            raw_df['Class'] = raw_df['Pclass'].map({1:'1st Class',2:'2nd Class',3:'3rd Class'})
            fig_fare = px.box(
                raw_df, x='Class', y='Fare', color='Class',
                color_discrete_sequence=['#10b981','#f59e0b','#ef4444'],
                title="Fare Distribution by Class",
                template='plotly_dark'
            )
            fig_fare.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                yaxis=dict(range=[0,300], gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=50, b=20, l=10, r=10)
            )
            with st.container(border=True):
                st.plotly_chart(fig_fare, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS & METRICS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.write("")
    st.markdown("#### 🔬 Model Diagnostics & Performance")

    # ── Metric cards ──────────────────────────────────────────
    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
    cards = [
        (col_v1, "Accuracy",  f"{metrics['accuracy']:.2%}",  "#818cf8"),
        (col_v2, "Precision", f"{metrics['precision']:.2%}", "#f8fafc"),
        (col_v3, "Recall",    f"{metrics['recall']:.2%}",    "#f8fafc"),
        (col_v4, "F1-Score",  f"{metrics['f1']:.2%}",        "#f8fafc"),
        (col_v5, "ROC-AUC",   f"{metrics['auc']:.2%}",       "#10b981"),
    ]
    for col, label, value, color in cards:
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="metric-value" style="color:{color};text-align:center">{value}</div>'
                    f'<div class="metric-label" style="text-align:center">{label}</div>',
                    unsafe_allow_html=True
                )

    st.write("")

    col_left, col_right = st.columns([1.3, 0.7], gap="large")

    # ── Feature Importance chart ──────────────────────────────
    with col_left:
        st.markdown("##### 🏆 Feature Decision Weightings")
        if 'feature_importances' in metrics:
            imp_df  = pd.DataFrame(metrics['feature_importances'])
            top_imp = imp_df.head(14).sort_values('importance', ascending=True)

            friendly = {
                'Sex_female':'Gender: Female','Sex_male':'Gender: Male',
                'Title_Mr':'Title: Mr.','Pclass_3':'3rd Class Ticket',
                'Fare':'Ticket Fare','AgeClass':'Age × Class Interaction',
                'Age':'Passenger Age','FarePerPerson':'Fare Per Person',
                'FamilySize':'Family Size','Title_Miss':'Title: Miss.',
                'Title_Mrs':'Title: Mrs.','Pclass_1':'1st Class Ticket',
                'Deck_U':'Deck: Unknown','SibSp':'Siblings/Spouses Aboard',
                'Pclass_2':'2nd Class Ticket','Title_Master':'Title: Master.',
                'IsAlone':'Traveling Alone','Embarked_S':'Embarked: Southampton'
            }
            top_imp['Feature'] = top_imp['feature'].map(lambda x: friendly.get(x, x))

            fig_imp = px.bar(
                top_imp, x='importance', y='Feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(title="Importance (Gain)", gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=20, b=20, l=10, r=10),
                height=420
            )
            with st.container(border=True):
                st.plotly_chart(fig_imp, use_container_width=True)

    # ── CV Benchmark table ────────────────────────────────────
    with col_right:
        st.markdown("##### 📋 Cross-Validation Benchmarks")
        st.caption("Stratified 5-Fold CV across all candidate models.")

        bench_records = []
        for name, bm in metrics['benchmarks'].items():
            bench_records.append({
                'Model': name,
                'CV Accuracy':  f"{bm['accuracy']:.2%}"  if not pd.isna(bm['accuracy'])  else "—",
                'CV Precision': f"{bm['precision']:.2%}" if not pd.isna(bm['precision']) else "—",
                'CV ROC-AUC':   f"{bm['auc']:.2%}"       if not pd.isna(bm['auc'])       else "—",
            })

        st.dataframe(pd.DataFrame(bench_records), hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — BATCH PREDICTIONS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.write("")
    st.markdown("#### 📁 Batch Passenger Predictions")

    col_b1, col_b2 = st.columns([1, 1.2], gap="large")

    with col_b1:
        with st.container(border=True):
            st.markdown("**📤 Upload Passenger Manifest CSV**")
            st.caption("Required columns: `Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked`")

            if raw_df is not None:
                template_csv = raw_df.drop(columns=['Survived','PassengerId'], errors='ignore').head(5).to_csv(index=False)
                st.download_button(
                    "📥 Download Sample Template",
                    data=template_csv,
                    file_name="titanic_template.csv",
                    mime="text/csv"
                )

            st.write("")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv", label_visibility="collapsed")

    with col_b2:
        if uploaded_file is not None:
            try:
                batch_input = pd.read_csv(uploaded_file)
                required = ['Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']
                missing  = [c for c in required if c not in batch_input.columns]

                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    batch_data = batch_input.copy()
                    if 'PassengerId' not in batch_data.columns:
                        batch_data['PassengerId'] = range(1, len(batch_data)+1)
                    if 'Ticket' not in batch_data.columns:
                        batch_data['Ticket'] = '12345'
                    if 'Cabin' not in batch_data.columns:
                        batch_data['Cabin'] = np.nan

                    preds  = pipeline.predict(batch_data)
                    probas = pipeline.predict_proba(batch_data)[:, 1]

                    batch_data['Predicted_Survived']  = preds
                    batch_data['Survival_Probability'] = np.round(probas * 100, 1)
                    batch_data['Survival_Status']      = pd.Series(preds).map({0:'Perished', 1:'Survived'})

                    survived_n  = int((preds == 1).sum())
                    perished_n  = int((preds == 0).sum())

                    st.success(f"✅ {len(batch_data)} passengers processed — **{survived_n} survive**, **{perished_n} perish**")

                    # Donut chart
                    fig_pie = go.Figure(go.Pie(
                        labels=['Survived','Perished'],
                        values=[survived_n, perished_n],
                        hole=0.45,
                        marker_colors=['#10b981','#f43f5e'],
                        textinfo='label+percent'
                    ))
                    fig_pie.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color':'#94a3b8', 'family':'Plus Jakarta Sans'},
                        height=240,
                        margin=dict(l=10,r=10,t=10,b=10),
                        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("**Preview (first 10 rows)**")
                    st.dataframe(
                        batch_data[['Name','Sex','Age','Pclass','Survival_Status','Survival_Probability']].head(10),
                        hide_index=True, use_container_width=True
                    )

                    result_csv = batch_data.drop(columns=['Class','Status'], errors='ignore').to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Predictions CSV",
                        data=result_csv,
                        file_name="titanic_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("⬆️ Upload a passenger CSV file on the left to run batch predictions.")
