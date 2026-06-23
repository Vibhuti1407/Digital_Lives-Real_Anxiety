import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests  
import uuid

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Digital Lives: Real Anxiety",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- GOOGLE ANALYTICS INTEGRATION ---
import streamlit.components.v1 as components

# --- BACKEND GOOGLE ANALYTICS (GA4 MEASUREMENT PROTOCOL) ---
def track_page_view():
    # Replace with the Secret Value you generated in Step 1
    api_secret = "-b3Sb5E-QEKVYE9wrMJCnA" 
    measurement_id = "G-SJ3PH3M4F5"
    
    # Track unique user sessions locally using streamlit session state
    if "ga_client_id" not in st.session_state:
        st.session_state.ga_client_id = str(uuid.uuid4())
        
        # Build the payload to hit Google's servers directly
        url = f"https://www.google-analytics.com/mp/collect?measurement_id={measurement_id}&api_secret={api_secret}"
        payload = {
            "client_id": st.session_state.ga_client_id,
            "events": [{
                "name": "page_view",
                "params": {
                    "page_title": "Digital Lives: Real Anxiety",
                    "engagement_time_msec": "1",
                    "debug_mode": 1
                }
            }]
        }
        try:
            requests.post(url, json=payload, timeout=2)
        except Exception:
            pass # Silently pass if there's a temporary network hiccup

# Execute the tracker instantly
track_page_view()

# --- CUSTOM CSS FOR BETTER AESTHETICS ---
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-top: 1rem !important;
        }
        .stApp h1 {
            font-size: 28px !important; /* Scale down the title for small screens */
            padding-bottom: 1rem !important;
        }
        .stMetric {
            margin-bottom: 10px !important;
        }
    }
    
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 100%;
    }
    .main { background-color: #f8f9fa; }
    .stApp h1 {
        color: white;         /* Change the text color */
        font-size: 50px;     /* Change the font size */
        font-family: Arial;  /* Change the font family */
        text-align: left;  /* Center the title */
        padding-bottom: 2.5rem;
    }
    .stMetric { background-color: black; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #d1d5db; }
    .stTabs [data-baseweb="tab-list"] button {
        margin-right: 15px; /* Adds space to the right of each tab */
        padding: 10px 20px; /* Increases internal padding (makes tabs taller/wider) */
    }
     .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px; 
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD CONNECTION TO TRAINED MODEL (.pkl) ---
@st.cache_resource
def load_trained_assets():
    try:
        with open('anxiety_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

assets = load_trained_assets()

# --- DATA PROCESSING ---
@st.cache_data
def load_and_clean():
    try:
        df = pd.read_csv('Survey_Responses.csv', encoding='latin1')
    except:
        df = pd.read_csv('Survey_Responses.csv', encoding='cp1252')
    
    # Mapping
    hour_map = {'0-2 hrs': 1, '2-5 hrs': 2, '5-8 hrs': 3, '8+ hrs': 4}
    df['daily_hours_numeric'] = df['daily_hours'].map(hour_map)
    
    # Formula Implementation (Score 1-10)
    df['daily_hours_scaled'] = df['daily_hours_numeric'] * 1.25
    df['digital_distress_score'] = ((df['anxiety_score'] + df['weak_focus'] + df['daily_hours_scaled']) / 15) * 10
    return df, hour_map

df, hour_map = load_and_clean()

# --- SIDEBAR: PERSONAL SCORE CALCULATION ---
with st.sidebar:
    total_responses = len(df)
    drained_pct = (df['drained_feeling'] == 'Yes').mean() * 100
    avg_anxiety = df['anxiety_score'].mean()

    st.markdown("""
        <div style="text-align: center; padding-bottom: 10px;">
            <h1 style='color: #B2BEB5; font-size: 28px; margin-bottom: 0;'>🧠 Digital Lives: Real Anxiety</h1>
            <p style='color: #94A3B8; font-size: 14px;'>Digital Impact Analysis Engine</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Insight Engine Status (Real-time Survey Metadata)
    st.markdown("### 📊 INSIGHT ENGINE")
    
    # Using real metrics from the uploaded CSV
    st.success(f"● Dataset: Survey_Responses.csv")
    st.info(f"● Sample Size: {total_responses} Users")
    st.warning(f"● Exhaustion Rate: {drained_pct:.1f}%")
    st.error(f"● Avg Anxiety Score: {avg_anxiety:.1f}/5")
    
    st.caption("Last Analysis: Feb 2026")
    st.divider()


# --- MAIN DASHBOARD ---
st.title("🧠 Digital Lives: Real Anxiety")

# TABBED INTERFACE
tab_overview, tab_viz, tab_model, tab_qual = st.tabs(["🏠 Home","📊 Data Trends", "🔍 Root Cause Analysis", "📝 Prdection Engine"], width="stretch")

with tab_overview:
    column1, column2 = st.columns([0.6,0.4])

    with column1: 
        # Section 1: Project Overview
        st.markdown(f"""
        <h4></h4>
        <h3>The Mission</h3>
        <p style='font-size:20px; color:gray;'>This study investigates the correlation between digital technology use and mental health indicators among adolescents. Using data collected from a target demographic (primarily ages 18–25), the research identifies how screen time, platform choice, and digital habits like "FOMO" contribute to self-reported anxiety and focus levels. Preliminary findings suggest a strong link between high daily usage (5+ hours) and increased anxiety scores.<p>
        <h3></h3>
        <h3>Overview of Data</h3>
        """, unsafe_allow_html=True)
        m1, m2= st.columns(2)
        m1.metric("Total Participants", len(df))
        m2.metric("Avg. Distress Score", f"{df['digital_distress_score'].mean():.2f}")
        m3, m4 = st.columns(2)
        m3.metric("Most Anxious Group", "Instagram Users")
        m4.metric("Key Root Cause", "Check Frequency")
      
    with column2:
        # Section 1: Image
        st.markdown("""
            <style>
                /* Target the div containing the image using data-testid for better stability */
                div[data-testid="stImage"] {
                    /*padding: 20px;  Adds 20px padding to all sides */
                    /* Or specify individual sides: */
                    padding-top: 0px;
                    padding-bottom: 10px; 
                    padding-left: 100px;
                    padding-right: 0px; 
                    width: 290;
                    height:260;
                }
             </style>
             """, unsafe_allow_html=True)
        st.image('home_img.png', use_container_width=True)


with tab_viz:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Average Distress by Platform")
        # Calculate the average score for each platform
        platform_avg = df.groupby('primary_platform')['digital_distress_score'].mean().reset_index()
        
        # Standard Bar Chart
        fig_p = px.bar(
            platform_avg, 
            x='primary_platform', 
            y='digital_distress_score', 
            color='primary_platform',
            labels={'digital_distress_score': 'Avg Distress Score', 'primary_platform': 'Platform'},
            template="simple_white"  # Makes it look like a standard research graph
        )
        st.plotly_chart(fig_p, use_container_width=True)
    
    with col_b:
        st.subheader("Root Cause: Checking Frequency")
        fig_s = px.scatter(df, x='check_frequency', y='digital_distress_score', 
                          size='anxiety_score', color='fomo_feelings',
                          labels={'check_frequency': 'Phone Checks / Hour'})
        st.plotly_chart(fig_s, use_container_width=True)

    # Adding a standard Trend Chart for screen time
    st.subheader("The Screen Time Trend")
    hours_trend = df.groupby('daily_hours')['digital_distress_score'].mean().reset_index()
    order = {"daily_hours": ["0-2 hrs", "2-5 hrs", "5-8 hrs", "8+ hrs"]}
    fig_line = px.line(
        hours_trend, 
        x='daily_hours', 
        y='digital_distress_score', 
        markers=True,
        category_orders=order,
        title="Impact of Daily Usage on Mental Health",
        template="simple_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)


with tab_model:
    st.header("Feature Importance (AI Engine)")
    st.write("Using a Random Forest Regressor to identify what actually drives digital anxiety.")
    
    # Simple Local Model
    le = LabelEncoder()
    model_df = df.copy()
    for col in ['fomo_feelings', 'social_comparision', 'morning_habit']:
        model_df[col] = le.fit_transform(model_df[col].astype(str))
    
    feats = ['daily_hours_numeric', 'check_frequency', 'fomo_feelings', 'social_comparision']
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(model_df[feats], model_df['anxiety_score'])
    
    importance = pd.DataFrame({'Feature': feats, 'Importance': rf.feature_importances_}).sort_values('Importance')
    
    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Reds')
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("""
    > **Research Conclusion:** The model identifies **Check Frequency** and **FOMO** as higher predictors of anxiety than total screen time alone. This suggests that *interruption* is more harmful than *duration*.
    """)

with tab_qual:
    def create_gauge(df):
        # 1. Map Daily Hours to Numeric Values
        hour_map = {
            '0-2 hrs': 1,
            '2-5 hrs': 2,
            '5-8 hrs': 3,
            '8+ hrs': 4
        }
        mapped_hours = df['daily_hours'].map(hour_map)
        
        # 2. Apply your specific formula to the averages
        avg_anxiety = df['anxiety_score'].mean()
        avg_focus = df['weak_focus'].mean()
        avg_hours = mapped_hours.mean()
        
        # Formula: (Anxiety + Focus + (Hours * 1.25) / 15) * 10
        # Note: This effectively creates a scale where ~100 is maximum impact
        strain_score = (avg_anxiety + avg_focus + (avg_hours * 1.25) / 15) * 10
        
        # 3. Create Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = strain_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Digital Strain Index", 'font': {'size': 22, 'color': 'white'}},
            number = {'font': {'color': 'white'}, 'suffix': "%"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ef4444"}, # Red bar for 'Strain'
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 40], 'color': '#10b981'},  # Low Impact (Green)
                    {'range': [40, 70], 'color': '#f59e0b'}, # Moderate Impact (Yellow)
                    {'range': [70, 100], 'color': '#ef4444'} # High Impact (Red)
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': strain_score}
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"},
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig


    
    col1, col2 = st.columns(2)

    with col1:
        st.header("AI Prediction Engine")
        if assets:
            st.write("Using trained `anxiety_model.pkl`")
            
            # Inputs using the same categories as training
            in_hrs = st.selectbox("Daily Screen Time", options=list(assets['hour_map'].keys()))
            in_check = st.number_input("Checks Per Hour", 0, 50, 5)
            in_fomo = st.selectbox("FOMO Feelings", options=assets['encoders']['fomo_feelings'].classes_)
            in_social = st.selectbox("Social Comparison", options=assets['encoders']['social_comparision'].classes_)
            in_morning = st.selectbox("Morning Habit", options=assets['encoders']['morning_habit'].classes_)
            
            analyze_bt = st.button("Predict Anxiety Score")
        else:
            st.error("Model file `anxiety_model.pkl` not found. Please run `train_model.py` first.")

    with col2:
        if analyze_bt:
            # Transform inputs using saved encoders
            hrs_val = assets['hour_map'][in_hrs]
            fomo_val = assets['encoders']['fomo_feelings'].transform([in_fomo])[0]
            social_val = assets['encoders']['social_comparision'].transform([in_social])[0]
            morning_val = assets['encoders']['morning_habit'].transform([in_morning])[0]
            
            # Predict
            X_input = np.array([[hrs_val, in_check, fomo_val, social_val, morning_val]])
            pred_score = assets['model'].predict(X_input)[0]
            
            if pred_score > 4.0:
                impact_level = "High"
            elif pred_score < 1.5:
                impact_level = "Low"
            else:
                impact_level = "Medium"

            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Predicted Anxiety (1-5)", f"{pred_score:.2f}")
            m_col2.metric("Impact Level", impact_level)

            st.divider()

            st.plotly_chart(create_gauge(df), width='stretch')



st.divider()

st.caption("Developed for Research Project | Topic: Digital Lives: Real Anxiety")


