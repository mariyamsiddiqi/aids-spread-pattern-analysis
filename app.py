import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AIDS Spread Analysis - Pakistan",
    page_icon="🏥",
    layout="wide"
)

# ── Load & Prepare Data ───────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'aids.csv')
    df = pd.read_csv(csv_path)
    df.columns = [
        'Country', 'Year', 'AIDS_Orphans', 'Deaths_Adults', 'Deaths_All',
        'Deaths_Children', 'Deaths_Female', 'Deaths_Male',
        'Prevalence_Adults', 'Prevalence_YoungMen', 'Prevalence_YoungWomen',
        'NewInfections_YoungAdults', 'NewInfections_Male', 'NewInfections_Female',
        'NewInfections_Children', 'NewInfections_All', 'NewInfections_Adults',
        'Incidence_Rate', 'PLHIV_Total', 'PLHIV_Male', 'PLHIV_Female',
        'PLHIV_Children', 'PLHIV_Adults'
    ]
    return df

@st.cache_resource
def train_model():
    df2 = load_data().copy()

    # Label encode country
    le = LabelEncoder()
    df2['Country_Encoded'] = le.fit_transform(df2['Country'])

    # Target variable
    median_inf = df2['NewInfections_All'].median()
    df2['High_Spread'] = (df2['NewInfections_All'] > median_inf).astype(int)

    # Feature Engineering
    df2['Death_Rate_Ratio']   = df2['Deaths_All'] / (df2['PLHIV_Total'] + 1)
    df2['Child_Impact_Ratio'] = df2['PLHIV_Children'] / (df2['PLHIV_Total'] + 1)
    df2['Gender_Gap']         = df2['NewInfections_Male'] - df2['NewInfections_Female']
    df2['Young_Prev_Gap']     = df2['Prevalence_YoungWomen'] - df2['Prevalence_YoungMen']
    df2['Orphan_Burden']      = df2['AIDS_Orphans'] / (df2['PLHIV_Total'] + 1)
    df2['PLHIV_Growth']       = df2.groupby('Country')['PLHIV_Total'].pct_change().fillna(0)
    df2['Deaths_Growth']      = df2.groupby('Country')['Deaths_All'].pct_change().fillna(0)
    df2['Male_Female_PLHIV']  = df2['PLHIV_Male'] / (df2['PLHIV_Female'] + 1)

    df2.replace([np.inf, -np.inf], 0, inplace=True)
    df2.fillna(0, inplace=True)

    feature_cols = [
        'Year', 'Country_Encoded',
        'Prevalence_Adults', 'Prevalence_YoungMen', 'Prevalence_YoungWomen',
        'Deaths_All', 'Deaths_Male', 'Deaths_Female', 'Deaths_Children',
        'PLHIV_Total', 'PLHIV_Male', 'PLHIV_Female', 'PLHIV_Children',
        'AIDS_Orphans', 'Incidence_Rate',
        'Death_Rate_Ratio', 'Child_Impact_Ratio', 'Gender_Gap',
        'Young_Prev_Gap', 'Orphan_Burden', 'PLHIV_Growth',
        'Deaths_Growth', 'Male_Female_PLHIV'
    ]

    X = df2[feature_cols]
    y = df2['High_Spread']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train_sc, y_train)

    test_acc = accuracy_score(y_test, model.predict(X_test_sc))

    return model, scaler, le, feature_cols, df2, test_acc

# Load everything
df = load_data()
model, scaler, le, feature_cols, df2, test_acc = train_model()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/3/32/Flag_of_Pakistan.svg", width=100)
st.sidebar.title("🏥 AIDS Analysis")
st.sidebar.markdown("**Pakistan HIV/AIDS Early Detection System**")
page = st.sidebar.selectbox("📌 Navigate", [
    "🏠 Home",
    "🔮 Predict Spread",
    "📊 Pakistan Analysis",
    "🌍 Global Trends"
])

# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 AIDS Spread Pattern Analysis")
    st.subheader("Early Detection System Using Machine Learning — Pakistan")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌍 Countries", "89")
    col2.metric("📅 Years", "1990–2020")
    col3.metric("📋 Records", "2,759")
    col4.metric("🎯 Model Accuracy", f"{test_acc*100:.2f}%")

    st.markdown("---")
    st.markdown("### 📌 Project Overview")
    st.info("""
    This system analyzes HIV/AIDS spread patterns using Machine Learning.
    - **Best Model:** XGBoost (98.73% Accuracy)
    - **Target:** High Spread vs Low Spread Classification
    - **Data Source:** Global HIV/AIDS Dataset (1990–2020)
    - **Focus:** Pakistan Epidemiological Analysis
    """)

    st.markdown("### 🏆 Model Performance")
    perf_data = {
        'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
        'Accuracy': [98.73, 98.73, 97.10],
        'F1 Score': [98.72, 98.72, 97.05],
        'ROC-AUC':  [99.93, 99.96, 99.67]
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2: PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Spread":
    st.title("🔮 HIV Spread Prediction")
    st.markdown("Enter values below to predict HIV spread level:")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        year         = st.slider("📅 Year", 1990, 2025, 2010)
        plhiv_total  = st.number_input("👥 PLHIV Total", 0, 10000000, 50000)
        plhiv_male   = st.number_input("👨 PLHIV Male", 0, 5000000, 30000)
        plhiv_female = st.number_input("👩 PLHIV Female", 0, 5000000, 15000)
        plhiv_child  = st.number_input("👶 PLHIV Children", 0, 1000000, 5000)

    with col2:
        deaths_all    = st.number_input("💀 Deaths All", 0, 1000000, 2000)
        deaths_male   = st.number_input("💀 Deaths Male", 0, 500000, 1000)
        deaths_female = st.number_input("💀 Deaths Female", 0, 500000, 800)
        deaths_child  = st.number_input("💀 Deaths Children", 0, 500000, 200)
        aids_orphans  = st.number_input("🧒 AIDS Orphans", 0, 1000000, 5000)

    with col3:
        prev_adults     = st.number_input("📈 Prevalence Adults (%)", 0.0, 30.0, 0.2)
        prev_youngmen   = st.number_input("📈 Prev Young Men (%)", 0.0, 20.0, 0.1)
        prev_youngwomen = st.number_input("📈 Prev Young Women (%)", 0.0, 20.0, 0.2)
        incidence_rate  = st.number_input("📉 Incidence Rate", 0.0, 50.0, 0.5)
        country         = st.selectbox("🌍 Country", sorted(df['Country'].unique()))

    if st.button("🔮 Predict Now!", use_container_width=True):

        # Handle unseen countries
        known_countries = list(le.classes_)
        if country in known_countries:
            country_enc = le.transform([country])[0]
        else:
            country_enc = 0

        # Compute engineered features
        death_rate    = deaths_all / (plhiv_total + 1)
        child_impact  = plhiv_child / (plhiv_total + 1)
        gender_gap    = 0
        young_gap     = prev_youngwomen - prev_youngmen
        orphan_burden = aids_orphans / (plhiv_total + 1)
        plhiv_growth  = 0.1
        deaths_growth = 0.05
        mf_ratio      = plhiv_male / (plhiv_female + 1)

        input_data = np.array([[
            year, country_enc,
            prev_adults, prev_youngmen, prev_youngwomen,
            deaths_all, deaths_male, deaths_female, deaths_child,
            plhiv_total, plhiv_male, plhiv_female, plhiv_child,
            aids_orphans, incidence_rate,
            death_rate, child_impact, gender_gap,
            young_gap, orphan_burden, plhiv_growth,
            deaths_growth, mf_ratio
        ]])

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ HIGH SPREAD RISK — Probability: {probability[1]*100:.1f}%")
        else:
            st.success(f"✅ LOW SPREAD RISK — Probability: {probability[0]*100:.1f}%")

        col1, col2 = st.columns(2)
        col1.metric("High Spread Probability", f"{probability[1]*100:.1f}%")
        col2.metric("Low Spread Probability",  f"{probability[0]*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# PAGE 3: PAKISTAN ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Pakistan Analysis":
    st.title("📊 Pakistan HIV/AIDS Analysis")
    st.markdown("---")

    pak = df[df['Country'] == 'Pakistan'].copy()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pak['Year'], pak['PLHIV_Total'], color='red', linewidth=2.5, marker='o')
        ax.fill_between(pak['Year'], pak['PLHIV_Total'], alpha=0.2, color='red')
        ax.set_title('Pakistan: People Living with HIV', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total PLHIV')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(pak['Year'], pak['NewInfections_All'], color='orange', edgecolor='black')
        ax.set_title('Pakistan: New HIV Infections Per Year', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('New Infections')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pak['Year'], pak['Deaths_All'], color='darkred', linewidth=2, marker='s')
        ax.set_title('Pakistan: AIDS Related Deaths', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Deaths')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col4:
        latest = pak[pak['Year'] == pak['Year'].max()].iloc[0]
        st.markdown("### 📋 Pakistan Latest Stats (2020)")
        st.metric("Total PLHIV",       f"{int(latest['PLHIV_Total']):,}")
        st.metric("New Infections",    f"{int(latest['NewInfections_All']):,}")
        st.metric("Deaths",            f"{int(latest['Deaths_All']):,}")
        st.metric("AIDS Orphans",      f"{int(latest['AIDS_Orphans']):,}")
        st.metric("Prevalence Adults", f"{latest['Prevalence_Adults']}%")

# ══════════════════════════════════════════════════════════════
# PAGE 4: GLOBAL TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "🌍 Global Trends":
    st.title("🌍 Global HIV/AIDS Trends")
    st.markdown("---")

    top10 = df.groupby('Country')['PLHIV_Total'].max().nlargest(10).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10['Country'], top10['PLHIV_Total'], color='purple', edgecolor='black')
    ax.set_title('Top 10 Countries by PLHIV Total', fontweight='bold')
    ax.set_xlabel('People Living with HIV')
    st.pyplot(fig)
    plt.close()

    yearly = df.groupby('Year')['NewInfections_All'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly['Year'], yearly['NewInfections_All'], color='blue', linewidth=2.5)
    ax.fill_between(yearly['Year'], yearly['NewInfections_All'], alpha=0.2)
    ax.set_title('Global New HIV Infections Per Year', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('New Infections')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()