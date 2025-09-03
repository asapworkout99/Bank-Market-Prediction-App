import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .likelihood-high {
        background: linear-gradient(90deg, #66bb6a 0%, #43a047 100%);
    }
    .likelihood-medium {
        background: linear-gradient(90deg, #ffa726 0%, #fb8c00 100%);
    }
    .likelihood-low {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .customer-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
    }
    .prediction-result {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=30)
def call_api(endpoint, method="GET", data=None):
    """Make API calls with error handling and caching"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json(), True
        else:
            st.error(f"API Error: {response.status_code}")
            return None, False
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API Server not running. Please start the FastAPI server first.")
        return None, False
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, False

def create_sample_data_offline():
    """Create sample bank marketing data when API is not available"""
    np.random.seed(42)
    customers = []
    jobs = ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student']
    marital_status = ['married', 'divorced', 'single']
    education_levels = ['unknown', 'secondary', 'primary', 'tertiary']
    
    for i in range(50):
        prob = np.random.beta(2, 5)  # Realistic subscription probability distribution
        likelihood = "High Likelihood" if prob > 0.7 else "Medium Likelihood" if prob > 0.4 else "Low Likelihood"
        customers.append({
            "customer_id": 1000 + i,
            "subscription_probability": prob,
            "likelihood": likelihood,
            "confidence_score": int(prob * 100),
            "age": np.random.randint(18, 75),
            "job": np.random.choice(jobs),
            "marital": np.random.choice(marital_status),
            "education": np.random.choice(education_levels),
            "balance": np.random.normal(1500, 3000),
            "duration": np.random.randint(0, 600),
            "campaign": np.random.randint(1, 10),
            "key_factors": ["Duration: " + str(np.random.randint(50, 500)) + " seconds", 
                           "Age group: " + ("young" if np.random.random() > 0.5 else "senior"),
                           "Job: " + np.random.choice(jobs[:3])]
        })
    return customers

def main():
    st.title("🏦 Bank Marketing Prediction Dashboard")
    st.markdown("### ML-Powered Term Deposit Subscription Prediction")
    
    # Check API health
    health_data, api_available = call_api("/health")
    
    if api_available:
        st.success(f"✅ Connected to API - Status: {health_data.get('status', 'unknown')}")
        model_info, _ = call_api("/")
        if model_info:
            st.info(f"Model Version: {model_info.get('model_version', 'N/A')} | "
                   f"Models Available: {', '.join(model_info.get('available_models', []))}")
    else:
        st.warning("⚠️ Running in offline mode with sample data")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose Section",
        ["📊 Executive Dashboard", "⚡ Single Prediction", "👥 Customer Analysis", 
         "📈 Batch Processing", "🔍 Model Insights", "⚙️ System Status"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Route to appropriate page
    if "Executive Dashboard" in page:
        show_executive_dashboard(api_available)
    elif "Single Prediction" in page:
        show_single_prediction(api_available)
    elif "Customer Analysis" in page:
        show_customer_analysis(api_available)
    elif "Batch Processing" in page:
        show_batch_processing(api_available)
    elif "Model Insights" in page:
        show_model_insights(api_available)
    elif "System Status" in page:
        show_system_status(api_available)

def show_executive_dashboard(api_available):
    st.header("📊 Executive Dashboard")
    
    # Get sample data - create offline predictions since batch endpoint doesn't exist yet
    customers = create_sample_data_offline()
    
    if not customers:
        st.error("No data available")
        return
    
    df = pd.DataFrame(customers)
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.markdown(f"""
        <div class="metric-container">
            <h3>Total Analyzed</h3>
            <h2>{total_customers:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_likelihood = len(df[df['likelihood'] == 'High Likelihood'])
        high_likelihood_pct = (high_likelihood / total_customers) * 100
        st.markdown(f"""
        <div class="metric-container likelihood-high">
            <h3>High Likelihood</h3>
            <h2>{high_likelihood} ({high_likelihood_pct:.1f}%)</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_subscription_prob = df['subscription_probability'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Subscription Rate</h3>
            <h2>{avg_subscription_prob:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        potential_subscribers = len(df[df['subscription_probability'] > 0.5])
        st.markdown(f"""
        <div class="metric-container likelihood-high">
            <h3>Potential Subscribers</h3>
            <h2>{potential_subscribers}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Likelihood Distribution")
        likelihood_counts = df['likelihood'].value_counts()
        colors = {'High Likelihood': '#2ecc71', 'Medium Likelihood': '#ffa726', 'Low Likelihood': '#ff4757'}
        fig_pie = px.pie(
            values=likelihood_counts.values,
            names=likelihood_counts.index,
            title="Customer Subscription Likelihood",
            color=likelihood_counts.index,
            color_discrete_map=colors,
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Subscription Probability Distribution")
        fig_hist = px.histogram(
            df, x='subscription_probability', nbins=25,
            title="Subscription Probability Distribution",
            labels={'subscription_probability': 'Subscription Probability', 'count': 'Customers'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text="Low")
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Medium")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="High")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Analysis by demographics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Analysis by Job Category")
        job_analysis = df.groupby('job')['subscription_probability'].agg(['mean', 'count']).reset_index()
        job_analysis = job_analysis.sort_values('mean', ascending=True)
        fig_job = px.bar(
            job_analysis, x='mean', y='job', orientation='h',
            title="Average Subscription Probability by Job",
            color='mean', color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_job, use_container_width=True)
    
    with col2:
        st.subheader("🎓 Analysis by Education Level")
        edu_analysis = df.groupby('education')['subscription_probability'].agg(['mean', 'count']).reset_index()
        fig_edu = px.bar(
            edu_analysis, x='education', y='mean',
            title="Subscription Probability by Education",
            color='mean', color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_edu, use_container_width=True)
    
    # Detailed insights
    st.subheader("💡 Campaign Insights")
    
    tab1, tab2, tab3 = st.tabs(["🎯 High-Value Prospects", "📊 Segment Analysis", "💡 Recommendations"])
    
    with tab1:
        high_likelihood_df = df[df['likelihood'] == 'High Likelihood'].head(10)
        if not high_likelihood_df.empty:
            st.markdown("**Top prospects for immediate contact:**")
            for idx, row in high_likelihood_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.write(f"**Customer #{row['customer_id']}**")
                        st.write(f"Likelihood: {row['likelihood']}")
                    with col2:
                        st.write(f"**Probability: {row['subscription_probability']:.1%}**")
                        st.write(f"Age: {row['age']} years")
                    with col3:
                        st.write(f"Job: {row['job']}")
                        st.write(f"Education: {row['education']}")
                    with col4:
                        if st.button("📞 Contact", key=f"contact_{idx}"):
                            st.success("Contact initiated!")
        else:
            st.info("🎉 All customers have been analyzed!")
    
    with tab2:
        st.subheader("Age Group Analysis")
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        age_analysis = df.groupby('age_group')['subscription_probability'].agg(['mean', 'count']).reset_index()
        fig_age = px.line(age_analysis, x='age_group', y='mean',
                         title="Subscription Probability by Age Group", markers=True)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with tab3:
        st.subheader("💡 Key Insights")
        insights = [
            f"📊 {high_likelihood_pct:.1f}% of customers have high subscription likelihood",
            f"💰 {potential_subscribers} customers are likely to subscribe",
            f"📈 Average subscription probability is {avg_subscription_prob:.1%}",
            f"🎯 Focus on {high_likelihood} high-likelihood prospects first",
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        st.subheader("🎯 Recommended Actions")
        recommendations = [
            "Contact high-likelihood customers within 48 hours",
            "Develop targeted campaigns for management and entrepreneur segments",
            "Focus on customers with longer call durations",
            "Implement follow-up strategies for medium-likelihood prospects"
        ]
        
        for rec in recommendations:
            st.markdown(f"✅ {rec}")

def show_single_prediction(api_available):
    st.header("⚡ Single Customer Prediction")
    
    # Get sample data first
    if api_available:
        sample_data, success = call_api("/sample-data")
        if success and sample_data:
            sample_customer = sample_data["sample_customer"]
        else:
            sample_customer = {
                "age": 35, "job": "management", "marital": "married", "education": "tertiary",
                "default": "no", "balance": 1500.5, "housing": "yes", "loan": "no",
                "contact": "cellular", "duration": 180, "campaign": 2, "pdays": -1,
                "previous": 0, "poutcome": "unknown", "month": "may"
            }
    else:
        sample_customer = {
            "age": 35, "job": "management", "marital": "married", "education": "tertiary",
            "default": "no", "balance": 1500.5, "housing": "yes", "loan": "no",
            "contact": "cellular", "duration": 180, "campaign": 2, "pdays": -1,
            "previous": 0, "poutcome": "unknown", "month": "may"
        }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Customer Information")
        
        with st.form("prediction_form"):
            st.markdown("##### Demographics")
            age = st.slider("Age", 18, 95, sample_customer.get("age", 35))
            job = st.selectbox("Job", 
                ["management", "technician", "entrepreneur", "blue-collar", "unknown", 
                 "retired", "admin.", "services", "self-employed", "unemployed", 
                 "housemaid", "student"], 
                index=0 if sample_customer.get("job") == "management" else 0)
            marital = st.selectbox("Marital Status", ["married", "divorced", "single"])
            education = st.selectbox("Education", ["unknown", "secondary", "primary", "tertiary"])
            
            st.markdown("##### Financial Information")
            balance = st.number_input("Account Balance ($)", 
                                    value=float(sample_customer.get("balance", 1500.5)), 
                                    step=100.0,
                                    help="Current account balance")
            default = st.selectbox("Has Credit Default?", ["no", "yes"])
            housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
            
            st.markdown("##### Contact Information")
            contact = st.selectbox("Contact Communication Type", 
                                 ["cellular", "telephone", "unknown"])
            duration = st.slider("Last Contact Duration (seconds)", 0, 1000, 
                                sample_customer.get("duration", 180))
            campaign = st.slider("Number of Contacts This Campaign", 1, 50, 
                                sample_customer.get("campaign", 2))
            
            st.markdown("##### Previous Campaign")
            pdays = st.number_input("Days Since Last Contact", -1, 999, 
                                  sample_customer.get("pdays", -1),
                                  help="-1 means never contacted")
            previous = st.slider("Number of Previous Contacts", 0, 50, 
                                sample_customer.get("previous", 0))
            poutcome = st.selectbox("Previous Campaign Outcome", 
                                  ["unknown", "success", "failure", "other"])
            
            st.markdown("##### Campaign Details")
            month = st.selectbox("Contact Month", 
                               ["jan", "feb", "mar", "apr", "may", "jun",
                                "jul", "aug", "sep", "oct", "nov", "dec"])
            
            predict_button = st.form_submit_button("🎯 Predict Subscription Likelihood", 
                                                 type="primary")
        
        if predict_button:
            # Prepare customer data according to API model
            customer_data = {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
                "month": month
            }
            
            if api_available:
                prediction, success = call_api("/predict", method="POST", data=customer_data)
                if success and prediction:
                    st.session_state.last_prediction = prediction
                else:
                    st.error("Prediction failed")
            else:
                # Offline prediction simulation
                mock_prob = np.random.beta(2, 5)
                likelihood = "High Likelihood" if mock_prob > 0.7 else "Medium Likelihood" if mock_prob > 0.4 else "Low Likelihood"
                prediction = {
                    "subscription_probability": mock_prob,
                    "likelihood": likelihood,
                    "confidence_score": int(mock_prob * 100),
                    "model_confidence": "High" if mock_prob > 0.8 or mock_prob < 0.2 else "Medium",
                    "key_factors": ["Duration: long call", "Job: management", "Age group: middle-aged"],
                    "model_version": "2.0.0"
                }
                st.session_state.last_prediction = prediction
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if hasattr(st.session_state, 'last_prediction') and st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            
            # Main prediction display
            likelihood = pred.get("likelihood", "Unknown")
            prob = pred.get("subscription_probability", 0.5)
            
            if "High" in likelihood:
                color = "green"
            elif "Medium" in likelihood:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2 style="color: {color};">🎯 {likelihood}</h2>
                <h3>Subscription Probability: {prob:.1%}</h3>
                <p><strong>Confidence Score:</strong> {pred.get('confidence_score', 50)}/100</p>
                <p><strong>Model Confidence:</strong> {pred.get('model_confidence', 'Medium')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Subscription Likelihood Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green" if prob > 0.7 else "orange" if prob > 0.4 else "red"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 50], 'color': "lightyellow"},
                        {'range': [50, 70], 'color': "lightblue"},
                        {'range': [70, 100], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 70}}))
            
            fig_gauge.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Key factors
            if pred.get('key_factors'):
                st.subheader("🔍 Key Factors")
                for factor in pred['key_factors']:
                    st.markdown(f"• {factor}")
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            if prob > 0.7:
                recommendations = [
                    "🎯 **HIGH PRIORITY**: Contact immediately",
                    "💰 Present premium term deposit options",
                    "📞 Schedule personal consultation",
                    "🎁 Offer competitive interest rates"
                ]
            elif prob > 0.4:
                recommendations = [
                    "📞 Follow up within 3-5 days",
                    "📊 Provide detailed product information",
                    "💡 Address specific customer concerns",
                    "📧 Send targeted marketing materials"
                ]
            else:
                recommendations = [
                    "⏰ Schedule for future campaigns",
                    "📈 Focus on relationship building",
                    "🎯 Include in general marketing pool",
                    "📋 Monitor for changing circumstances"
                ]
            
            for rec in recommendations:
                st.markdown(rec)
            
        else:
            st.info("👆 Enter customer information and click 'Predict Subscription Likelihood' to see results")
            
            # Show sample prediction for demo
            st.subheader("📋 Sample Customer Profile")
            st.markdown(f"""
            **Example Customer:**
            - Age: {sample_customer['age']} years
            - Job: {sample_customer['job']}
            - Balance: ${sample_customer['balance']:,.1f}
            - Contact Duration: {sample_customer['duration']} seconds
            - Campaign: {sample_customer['campaign']} contacts
            - Education: {sample_customer['education']}
            """)

def show_customer_analysis(api_available):
    st.header("👥 Customer Analysis & Segmentation")
    
    # Create sample data for analysis
    customers = create_sample_data_offline()
    
    if not customers:
        st.warning("No customer data available")
        return
    
    df = pd.DataFrame(customers)
    
    st.subheader(f"📊 Customer Portfolio Analysis ({len(customers)} customers)")
    
    # Segment customers by likelihood
    high_likelihood = df[df['likelihood'] == 'High Likelihood']
    medium_likelihood = df[df['likelihood'] == 'Medium Likelihood']
    low_likelihood = df[df['likelihood'] == 'Low Likelihood']
    
    tab1, tab2, tab3 = st.tabs([
        f"🎯 High Likelihood ({len(high_likelihood)})", 
        f"📊 Medium Likelihood ({len(medium_likelihood)})", 
        f"📈 Low Likelihood ({len(low_likelihood)})"
    ])
    
    with tab1:
        if not high_likelihood.empty:
            st.markdown("**Customers with highest subscription probability:**")
            for idx, row in high_likelihood.head(10).iterrows():
                with st.expander(f"Customer #{row['customer_id']} - {row['subscription_probability']:.1%} likelihood"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📊 Prediction Profile**")
                        st.write(f"Probability: {row['subscription_probability']:.1%}")
                        st.write(f"Confidence: {row['confidence_score']}/100")
                        st.write(f"Category: {row['likelihood']}")
                    
                    with col2:
                        st.markdown("**👤 Demographics**")
                        st.write(f"Age: {row['age']} years")
                        st.write(f"Job: {row['job']}")
                        st.write(f"Education: {row['education']}")
                    
                    with col3:
                        st.markdown("**🔍 Key Factors**")
                        for factor in row['key_factors'][:3]:
                            st.write(f"• {factor}")
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("📞 Priority Call", key=f"call_{idx}"):
                            st.success("Priority call scheduled!")
                    with col2:
                        if st.button("📧 Personal Email", key=f"email_{idx}"):
                            st.success("Personal email sent!")
                    with col3:
                        if st.button("🎁 Special Offer", key=f"offer_{idx}"):
                            st.success("Special offer prepared!")
                    with col4:
                        if st.button("📋 Add Note", key=f"note_{idx}"):
                            st.info("Note added to profile!")
        else:
            st.info("No high-likelihood customers in current sample")
    
    with tab2:
        if not medium_likelihood.empty:
            st.markdown("**Customers requiring targeted follow-up:**")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_prob = medium_likelihood['subscription_probability'].mean()
                st.metric("Average Probability", f"{avg_prob:.1%}")
            with col2:
                avg_age = medium_likelihood['age'].mean()
                st.metric("Average Age", f"{avg_age:.0f} years")
            with col3:
                avg_duration = medium_likelihood['duration'].mean()
                st.metric("Avg Call Duration", f"{avg_duration:.0f}s")
            
            # Show top medium-likelihood customers
            for idx, row in medium_likelihood.head(5).iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Customer #{row['customer_id']}**")
                        st.write(f"Likelihood: {row['subscription_probability']:.1%}")
                    with col2:
                        st.write(f"Age: {row['age']}, Job: {row['job']}")
                        st.write(f"Duration: {row['duration']}s")
                    with col3:
                        if st.button("📞 Follow-up", key=f"followup_{idx}"):
                            st.success("Follow-up scheduled!")
        else:
            st.info("No medium-likelihood customers in current sample")
    
    with tab3:
        if not low_likelihood.empty:
            st.markdown("**Customers for future nurturing campaigns:**")
            
            # Analysis of low-likelihood segment
            col1, col2 = st.columns(2)
            
            with col1:
                job_dist = low_likelihood['job'].value_counts()
                fig_jobs = px.bar(x=job_dist.values, y=job_dist.index, orientation='h',
                                title="Job Distribution - Low Likelihood Segment")
                st.plotly_chart(fig_jobs, use_container_width=True)
            
            with col2:
                age_dist = low_likelihood['age'].hist(bins=10)
                fig_age = px.histogram(low_likelihood, x='age', nbins=10,
                                     title="Age Distribution - Low Likelihood Segment")
                st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No low-likelihood customers in current sample")
    
    # Overall analysis
    st.subheader("📈 Portfolio Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job category analysis
        job_analysis = df.groupby('job')['subscription_probability'].agg(['mean', 'count']).reset_index()
        job_analysis = job_analysis.sort_values('mean', ascending=False)
        
        fig_job_perf = px.bar(job_analysis.head(8), x='job', y='mean',
                             title="Top Job Categories by Subscription Probability",
                             color='mean', color_continuous_scale='Greens')
        st.plotly_chart(fig_job_perf, use_container_width=True)
    
    with col2:
        # Age group analysis
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        age_analysis = df.groupby('age_group')['subscription_probability'].agg(['mean', 'count']).reset_index()
        
        fig_age_perf = px.line(age_analysis, x='age_group', y='mean',
                              title="Subscription Probability by Age Group",
                              markers=True, color_discrete_sequence=['#2E86AB'])
        st.plotly_chart(fig_age_perf, use_container_width=True)

def show_batch_processing(api_available):
    st.header("📈 Batch Processing & File Upload")
    
    st.subheader("📁 Upload Customer Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} customers found.")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Validate required columns based on bank marketing model
            required_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                           'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 
                           'previous', 'poutcome', 'month']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                
                # Show required format
                st.subheader("📝 Required CSV Format")
                if api_available:
                    sample_data, success = call_api("/sample-data")
                    if success and sample_data:
                        sample_df = pd.DataFrame([sample_data["sample_customer"]])
                        st.dataframe(sample_df)
                
            else:
                if st.button("🚀 Process Batch Predictions", type="primary"):
                    # Process in batches
                    batch_size = 50
                    all_predictions = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(0, len(df), batch_size):
                        batch = df.iloc[i:i+batch_size]
                        batch_data = batch.to_dict('records')
                        
                        if api_available:
                            batch_request = {"customers": batch_data}
                            result, success = call_api("/predict/batch", method="POST", data=batch_request)
                            if success and result:
                                all_predictions.extend(result.get('predictions', []))
                        else:
                            # Simulate batch processing
                            for _, row in batch.iterrows():
                                mock_prob = np.random.beta(2, 5)
                                likelihood = "High Likelihood" if mock_prob > 0.7 else "Medium Likelihood" if mock_prob > 0.4 else "Low Likelihood"
                                all_predictions.append({
                                    "customer_id": row.get('id', i),
                                    "subscription_probability": mock_prob,
                                    "likelihood": likelihood,
                                    "confidence_score": int(mock_prob * 100)
                                })
                        
                        progress = min((i + batch_size) / len(df), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing batch {i//batch_size + 1}...")
                    
                    # Results
                    if all_predictions:
                        results_df = pd.DataFrame(all_predictions)
                        st.success(f"✅ Processed {len(results_df)} customer predictions!")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            high_likelihood_count = len(results_df[results_df['likelihood'] == 'High Likelihood'])
                            st.metric("High Likelihood", high_likelihood_count, 
                                     f"{(high_likelihood_count/len(results_df))*100:.1f}%")
                        
                        with col2:
                            avg_prob = results_df['subscription_probability'].mean()
                            st.metric("Average Probability", f"{avg_prob:.1%}")
                        
                        with col3:
                            likely_subscribers = len(results_df[results_df['subscription_probability'] > 0.5])
                            st.metric("Likely Subscribers", likely_subscribers)
                        
                        # Results visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            likelihood_dist = results_df['likelihood'].value_counts()
                            fig_dist = px.pie(values=likelihood_dist.values, names=likelihood_dist.index,
                                            title="Prediction Distribution")
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            fig_hist = px.histogram(results_df, x='subscription_probability', nbins=20,
                                                  title="Subscription Probability Distribution")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Results table
                        st.subheader("📊 Prediction Results")
                        results_display = results_df.copy()
                        results_display['subscription_probability'] = results_display['subscription_probability'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(results_display)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download Results CSV",
                            data=csv,
                            file_name=f"bank_marketing_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    progress_bar.empty()
                    status_text.empty()
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file with customer data to get started")
        
        # Show sample format
        st.subheader("📋 Required CSV Format")
        if api_available:
            sample_data, success = call_api("/sample-data")
            if success and sample_data:
                sample_customer = sample_data["sample_customer"]
                sample_df = pd.DataFrame([sample_customer, sample_customer, sample_customer])
                sample_df.index = ['Customer 1', 'Customer 2', 'Customer 3']
                st.dataframe(sample_df)
                
                # Download sample
                sample_csv = sample_df.to_csv()
                st.download_button(
                    label="⬇️ Download Sample CSV Template",
                    data=sample_csv,
                    file_name="bank_marketing_sample.csv",
                    mime="text/csv"
                )

def show_model_insights(api_available):
    st.header("🔍 Model Insights & Performance")
    
    if api_available:
        # Get model information
        model_info, success1 = call_api("/model/info")
        features_info, success2 = call_api("/model/features")
        
        if success1 and model_info:
            st.subheader("📊 Model Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Version", model_info.get('metadata', {}).get('model_version', 'N/A'))
            with col2:
                st.metric("Total Features", model_info.get('total_features', 0))
            with col3:
                models_loaded = model_info.get('models_loaded', [])
                st.metric("Models Loaded", len(models_loaded))
            
            # Show available models
            if models_loaded:
                st.subheader("🤖 Available Models")
                for model in models_loaded:
                    st.markdown(f"• {model}")
            
            # Show some selected features
            if model_info.get('selected_features'):
                st.subheader("🎯 Key Features (Sample)")
                sample_features = model_info['selected_features'][:15]
                for i, feature in enumerate(sample_features, 1):
                    st.markdown(f"{i}. {feature}")
        
        if success2 and features_info:
            st.subheader("📈 All Model Features")
            all_features = features_info.get('features', [])
            if all_features:
                st.markdown(f"**Total Features Used:** {len(all_features)}")
                
                # Show features in columns
                col1, col2, col3 = st.columns(3)
                third = len(all_features) // 3
                
                with col1:
                    for feature in all_features[:third]:
                        st.markdown(f"• {feature}")
                with col2:
                    for feature in all_features[third:2*third]:
                        st.markdown(f"• {feature}")
                with col3:
                    for feature in all_features[2*third:]:
                        st.markdown(f"• {feature}")
    
    else:
        st.warning("⚠️ API not available. Showing sample model information.")
        
        # Mock model insights
        st.subheader("📊 Model Information (Sample)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Version", "2.0.0")
        with col2:
            st.metric("Total Features", "85")
        with col3:
            st.metric("Models Available", "XGBoost, LightGBM")
    
    # Feature importance simulation (since we don't have actual feature importance from API)
    st.subheader("🎯 Key Feature Categories")
    
    feature_categories = {
        "Contact Information": ["duration", "campaign", "contact", "month"],
        "Customer Demographics": ["age", "job", "marital", "education"],
        "Financial Profile": ["balance", "housing", "loan", "default"],
        "Previous Campaign": ["previous", "pdays", "poutcome"],
        "Derived Features": ["age_group", "balance_positive", "duration_long", "campaign_high"]
    }
    
    for category, features in feature_categories.items():
        with st.expander(f"📁 {category}"):
            for feature in features:
                st.markdown(f"• **{feature}**: Important predictor in the model")
    
    st.subheader("💡 Model Insights")
    insights = [
        "📞 **Call Duration**: Longer conversations indicate higher engagement and subscription likelihood",
        "🎯 **Campaign Contacts**: Too many contacts can reduce effectiveness",
        "👥 **Demographics**: Age and job type are strong predictors",
        "💰 **Financial Status**: Account balance and existing loans influence decisions",
        "📅 **Timing**: Month of contact affects campaign success",
        "🔄 **Previous Outcomes**: Past campaign results strongly predict future behavior"
    ]
    
    for insight in insights:
        st.markdown(insight)

def show_system_status(api_available):
    st.header("⚙️ System Status & Configuration")
    
    if api_available:
        # Get system status
        health_data, success1 = call_api("/health")
        root_data, success2 = call_api("/")
        
        if success1 and health_data:
            st.subheader("🔧 System Health")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = "🟢 Online" if health_data.get('models_loaded') else "🔴 Offline"
                st.markdown(f"**API Status:** {status}")
            
            with col2:
                timestamp = health_data.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    # Parse and format timestamp
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%H:%M:%S')
                        st.markdown(f"**Last Check:** {formatted_time}")
                    except:
                        st.markdown(f"**Last Check:** {timestamp}")
                else:
                    st.markdown(f"**Last Check:** {timestamp}")
            
            with col3:
                models = health_data.get('models_available', [])
                st.markdown(f"**Models:** {len(models)} active")
            
            with col4:
                st.markdown("**Status:** Healthy")
        
        if success2 and root_data:
            st.subheader("📊 API Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**API Details:**")
                st.markdown(f"• Model Version: {root_data.get('model_version', 'N/A')}")
                st.markdown(f"• Models Loaded: {root_data.get('model_loaded', False)}")
                st.markdown(f"• Feature Count: {root_data.get('feature_count', 'N/A')}")
            
            with col2:
                available_models = root_data.get('available_models', [])
                st.markdown("**Available Models:**")
                for model in available_models:
                    st.markdown(f"• {model}")
        
        # API endpoints
        st.subheader("🔗 Available API Endpoints")
        endpoints = [
            {"method": "GET", "endpoint": "/", "description": "Root endpoint with basic info"},
            {"method": "GET", "endpoint": "/health", "description": "Health check"},
            {"method": "POST", "endpoint": "/predict", "description": "Single customer prediction"},
            {"method": "POST", "endpoint": "/predict/batch", "description": "Batch predictions"},
            {"method": "GET", "endpoint": "/model/info", "description": "Model metadata"},
            {"method": "GET", "endpoint": "/model/features", "description": "Feature list"},
            {"method": "GET", "endpoint": "/sample-data", "description": "Sample customer data"}
        ]
        
        for ep in endpoints:
            st.markdown(f"• **{ep['method']}** `{ep['endpoint']}` - {ep['description']}")
    
    else:
        st.error("❌ API Server not available")
        st.markdown("""
        ### 🚀 To start the system:
        
        1. **Install dependencies:**
        ```bash
        pip install fastapi uvicorn streamlit pandas numpy scikit-learn xgboost lightgbm plotly requests joblib
        ```
        
        2. **Start the API server:**
        ```bash
        python fastapi_backend.py
        ```
        
        3. **Run this dashboard:**
        ```bash
        streamlit run streamlit_dashboard.py
        ```
        
        4. **API will be available at:** http://localhost:8000
        5. **Dashboard will be available at:** http://localhost:8501
        """)
    
    # Connection test
    st.subheader("🔍 Connection Test")
    if st.button("Test API Connection", type="primary"):
        test_result, success = call_api("/health")
        if success:
            st.success("✅ API connection successful!")
            st.json(test_result)
        else:
            st.error("❌ API connection failed")

if __name__ == "__main__":
    main()