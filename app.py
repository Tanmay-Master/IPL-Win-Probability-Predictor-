import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(page_title="IPL Win Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom styling with enhanced design
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    .main-header {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(45deg, #0f3460, #16213e);
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 48px;
        color: #00d4ff;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
    }
    
    .main-header p {
        font-size: 18px;
        color: #a8dadc;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .stButton>button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #0099cc, #00d4ff);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #264653, #2a9d8f);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        border-left: 5px solid #00d4ff;
    }
    
    .probability-card-batting {
        background: linear-gradient(135deg, #2a9d8f, #1db584);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(42, 157, 143, 0.3);
        border-top: 4px solid #00d4ff;
    }
    
    .probability-card-bowling {
        background: linear-gradient(135deg, #e76f51, #f4a261);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(231, 111, 81, 0.3);
        border-top: 4px solid #ff6b35;
    }
    
    .model-selector {
        background: linear-gradient(135deg, #0f3460, #16213e);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        border-left: 5px solid #00d4ff;
    }
    
    .input-section {
        background: linear-gradient(135deg, #16213e, #0f3460);
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #f4a261;
    }
    
    .section-header {
        color: #00d4ff;
        font-size: 24px;
        font-weight: bold;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #00d4ff;
    }
    
    .stCheckbox {
        margin: 10px 0;
    }
    
    .stNumberInput, .stSelectbox {
        margin: 10px 0;
    }
    
    .comparison-container {
        background: linear-gradient(135deg, #264653, #1a3a3a);
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid #2a9d8f;
    }
    
    .ensemble-box {
        background: linear-gradient(135deg, #2196F3, #1565C0);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.3);
        border-top: 4px solid #64B5F6;
        margin-top: 20px;
    }
    
    .stat-value {
        font-size: 42px;
        font-weight: bold;
        color: #00d4ff;
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 16px;
        color: #a8dadc;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .success-message {
        background: #2a9d8f;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border-left: 5px solid #00d4ff;
    }
    
    .warning-message {
        background: #e76f51;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border-left: 5px solid #ff6b35;
    }
    
    .error-message {
        background: #d32f2f;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border-left: 5px solid #ff1744;
    }
    
    .info-message {
        background: #0288d1;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        border-left: 5px solid #03a9f4;
    }
    
    .match-over-card {
        background: linear-gradient(135deg, #d32f2f, #f44336);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(211, 47, 47, 0.4);
        border-top: 5px solid #ff1744;
        margin: 30px 0;
    }
    
    .match-won-card {
        background: linear-gradient(135deg, #388e3c, #4caf50);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(56, 142, 60, 0.4);
        border-top: 5px solid #00e676;
        margin: 30px 0;
    }
    
    .analysis-table {
        background: linear-gradient(135deg, #264653, #2a9d8f);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame {
        background: #16213e;
    }
</style>
""", unsafe_allow_html=True)

# Teams and cities
teams = ['Royal Challengers Bangalore', 'Mumbai Indians', 'Kolkata Knight Riders',
         'Rajasthan Royals', 'Chennai Super Kings', 'Sunrisers Hyderabad',
         'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans']

cities = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Hyderabad', 'Chennai', 'Jaipur',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London',
          'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack',
          'Nagpur', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Bengaluru', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow', 'Guwahati',
          'Chandigarh', 'Dharamsala', 'Mohali']

# Load models
@st.cache_resource
def load_models():
    lr_pipe = pickle.load(open('lr_pipe.pkl', 'rb'))
    rf_pipe = pickle.load(open('rf_pipe.pkl', 'rb'))
    return lr_pipe, rf_pipe

lr_pipe, rf_pipe = load_models()

# Validation functions
def validate_overs_format(overs):
    """Validate overs format (e.g., 5.3 means 5 overs and 3 balls, not 5.3 overs)"""
    overs_int = int(overs)
    balls = round((overs - overs_int) * 10)
    
    if balls >= 6:
        return False, f"Invalid overs format! {overs_int}.{balls} - Balls in an over cannot be 6 or more. Did you mean {overs_int + 1}.{balls - 6}?"
    
    return True, None

def calculate_actual_balls(overs):
    """Convert overs to actual balls bowled"""
    overs_int = int(overs)
    balls = round((overs - overs_int) * 10)
    return (overs_int * 6) + balls

def validate_match_state(score, target, overs, wickets):
    """Comprehensive match state validation"""
    errors = []
    warnings = []
    
    # Check if all wickets are lost
    if wickets >= 10:
        return False, ["Match is already over! All 10 wickets have fallen. Bowling team wins."], []
    
    # Check if target is already achieved
    if score >= target:
        return False, [f"Match is already over! Target achieved. Batting team wins by {10 - wickets} wickets."], []
    
    # Check if overs are completed
    if overs >= 20.0:
        return False, ["Match is already over! 20 overs completed. Bowling team wins."], []
    
    # Validate overs format
    is_valid_over, over_error = validate_overs_format(overs)
    if not is_valid_over:
        errors.append(over_error)
    
    # Check if current run rate is reasonable
    if overs > 0:
        crr = score / overs
        if crr > 36:  # 36 runs per over is maximum possible (6 sixes)
            warnings.append(f"Current run rate ({crr:.2f}) seems unusually high. Please verify the score.")
        elif crr < 1 and overs > 5:
            warnings.append(f"Current run rate ({crr:.2f}) seems unusually low. Please verify the score.")
    
    # Check if score is reasonable for wickets lost
    if wickets > 0 and overs > 0:
        avg_score_per_wicket = score / wickets if wickets > 0 else score
        if avg_score_per_wicket < 5 and score > 20:
            warnings.append(f"Average score per wicket ({avg_score_per_wicket:.1f}) seems low. Verify wickets count.")
    
    # Check if target is reasonable for T20
    if target < 50:
        warnings.append("Target seems very low for a T20 match.")
    elif target > 300:
        warnings.append("Target seems very high for a T20 match.")
    
    # Check balls remaining
    actual_balls = calculate_actual_balls(overs)
    balls_left = 120 - actual_balls
    
    if balls_left <= 0:
        errors.append("No balls remaining! Match should be over.")
    
    # Check if target is achievable
    runs_needed = target - score
    wickets_left = 10 - wickets
    
    if balls_left > 0:
        required_rr = (runs_needed * 6) / balls_left
        
        if required_rr > 36:
            warnings.append(f"Required run rate ({required_rr:.2f}) is mathematically impossible. Bowling team likely to win.")
        elif required_rr > 20:
            warnings.append(f"Required run rate ({required_rr:.2f}) is extremely difficult to achieve.")
        
        # Check if enough wickets for aggressive batting
        if wickets_left <= 2 and required_rr > 12:
            warnings.append(f"Only {wickets_left} wickets remaining with high required run rate. Very difficult situation.")
    
    # Check for impossible score
    max_possible_score = actual_balls * 6  # Maximum 6 runs per ball
    if score > max_possible_score:
        errors.append(f"Impossible score! Maximum possible score in {overs} overs is {max_possible_score} runs.")
    
    return len(errors) == 0, errors, warnings

# Header
st.markdown("""
<div class="main-header">
    <h1>IPL WIN PROBABILITY PREDICTOR</h1>
    <p>Advanced Machine Learning Analysis for Cricket Match Predictions</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.markdown("### ABOUT THIS PREDICTOR")
    st.info("This advanced AI-powered system utilizes state-of-the-art machine learning algorithms to analyze real-time match dynamics and deliver accurate win probability predictions.")
    
    st.markdown("### AVAILABLE MODELS")
    st.markdown("""
    **Logistic Regression (LR)**
    - Linear classification model
    - Fast computation
    - High interpretability
    - Excellent for linear separable data
    
    **Random Forest (RF)**
    - Ensemble learning method
    - Captures non-linear patterns
    - Robust to outliers
    - Superior accuracy for complex relationships
    """)
    
    st.markdown("### KEY METRICS USED")
    st.markdown("""
    - Current Run Rate (CRR)
    - Required Run Rate (RRR)
    - Remaining Runs & Balls
    - Remaining Wickets
    - Team Strength Analysis
    - Venue Characteristics
    """)
    
    st.markdown("### INPUT GUIDELINES")
    st.markdown("""
    **Overs Format:**
    - Use decimal format: X.Y
    - X = completed overs
    - Y = balls in current over
    - Example: 5.3 = 5 overs + 3 balls
    - Invalid: 5.6 or higher (max 5 balls)
    
    **Valid Ranges:**
    - Overs: 0.1 to 19.5
    - Wickets: 0 to 9 (10 = match over)
    - Score: Must be less than target
    """)

# Model selection section
st.markdown('<div class="model-selector">', unsafe_allow_html=True)
st.markdown("### MODEL SELECTION")

col_model1, col_model2, col_model3 = st.columns(3)

with col_model1:
    use_lr = st.checkbox("Logistic Regression", value=True)
with col_model2:
    use_rf = st.checkbox("Random Forest", value=True)
with col_model3:
    show_comparison = st.checkbox("Comparison Mode")

st.markdown('</div>', unsafe_allow_html=True)

if not use_lr and not use_rf:
    st.markdown('<div class="warning-message"><strong>Alert:</strong> Please select at least one model to proceed!</div>', unsafe_allow_html=True)
    st.stop()

# Main content area
st.markdown("### TEAM SELECTION")
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", sorted(teams), key="bat_team")

with col2:
    bowling_team = st.selectbox("Bowling Team", sorted(teams), key="bowl_team")

# Validate team selection
if batting_team == bowling_team:
    st.markdown('<div class="warning-message"><strong>Invalid Selection:</strong> Batting and Bowling teams must be different!</div>', unsafe_allow_html=True)
else:
    # Venue selection
    city = st.selectbox("Match Venue", sorted(cities))
    
    # Target score
    target = st.number_input("Target Score", min_value=1, max_value=500, value=150, step=1, 
                            help="The target score batting team needs to chase")

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### CURRENT MATCH DETAILS")
    
    st.info("**Note:** Enter overs in format X.Y where X is completed overs and Y is balls (0-5). Example: 5.3 means 5 overs and 3 balls.")

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input("Current Score", min_value=0, max_value=500, value=0, step=1,
                               help="Current score of the batting team")

    with col4:
        overs = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, value=5.0, step=0.1,
                               help="Format: X.Y (X=overs, Y=balls 0-5). Example: 5.3 = 5 overs 3 balls")

    with col5:
        wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=2, step=1,
                                  help="Number of wickets lost (0-10)")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("GENERATE PREDICTION", key="predict_btn"):
        # Comprehensive validation
        is_valid, errors, warnings = validate_match_state(score, target, overs, wickets)
        
        # Display errors
        if errors:
            for error in errors:
                st.markdown(f'<div class="error-message"><strong>Error:</strong> {error}</div>', unsafe_allow_html=True)
        
        # Display warnings
        if warnings:
            for warning in warnings:
                st.markdown(f'<div class="warning-message"><strong>Warning:</strong> {warning}</div>', unsafe_allow_html=True)
        
        # Stop if there are critical errors
        if not is_valid:
            # Check specific game over conditions
            if wickets >= 10:
                st.markdown(f"""
                <div class="match-over-card">
                    <h2 style='color: white; margin-bottom: 20px;'>MATCH OVER</h2>
                    <h3 style='color: #ffeb3b;'>All Wickets Down!</h3>
                    <p style='font-size: 18px; margin-top: 20px;'>{batting_team} all out for {score} runs</p>
                    <h2 style='color: #00e676; margin-top: 30px;'>{bowling_team} WINS!</h2>
                    <p style='font-size: 16px; margin-top: 10px;'>Winning margin: {target - score - 1} runs</p>
                </div>
                """, unsafe_allow_html=True)
            elif score >= target:
                st.markdown(f"""
                <div class="match-won-card">
                    <h2 style='color: white; margin-bottom: 20px;'>MATCH WON</h2>
                    <h3 style='color: #ffeb3b;'>Target Achieved!</h3>
                    <p style='font-size: 18px; margin-top: 20px;'>{batting_team} scored {score}/{wickets}</p>
                    <h2 style='color: #00e676; margin-top: 30px;'>{batting_team} WINS!</h2>
                    <p style='font-size: 16px; margin-top: 10px;'>Winning margin: {10 - wickets} wickets</p>
                    <p style='font-size: 14px; margin-top: 5px;'>Balls remaining: {120 - calculate_actual_balls(overs)}</p>
                </div>
                """, unsafe_allow_html=True)
            elif overs >= 20.0:
                st.markdown(f"""
                <div class="match-over-card">
                    <h2 style='color: white; margin-bottom: 20px;'>MATCH OVER</h2>
                    <h3 style='color: #ffeb3b;'>All Overs Completed!</h3>
                    <p style='font-size: 18px; margin-top: 20px;'>{batting_team} scored {score}/{wickets}</p>
                    <h2 style='color: #00e676; margin-top: 30px;'>{bowling_team} WINS!</h2>
                    <p style='font-size: 16px; margin-top: 10px;'>Winning margin: {target - score - 1} runs</p>
                </div>
                """, unsafe_allow_html=True)
            st.stop()
        
        # Calculate metrics with actual balls
        actual_balls = calculate_actual_balls(overs)
        runs_left = target - score
        balls_left = 120 - actual_balls
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Create input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr],
            'total_runs_x': [score],
            'wickets_fallen': [wickets]
        })

        # Display match summary
        st.markdown("### MATCH SUMMARY")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            st.markdown(f'<div class="metric-card"><div class="stat-label">Current Run Rate</div><div class="stat-value">{crr:.2f}</div></div>', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f'<div class="metric-card"><div class="stat-label">Required Run Rate</div><div class="stat-value">{rrr:.2f}</div></div>', unsafe_allow_html=True)
        with col_s3:
            st.markdown(f'<div class="metric-card"><div class="stat-label">Runs Needed</div><div class="stat-value">{int(runs_left)}</div></div>', unsafe_allow_html=True)
        with col_s4:
            st.markdown(f'<div class="metric-card"><div class="stat-label">Balls Remaining</div><div class="stat-value">{int(balls_left)}</div></div>', unsafe_allow_html=True)

        # Additional match context
        st.markdown("### MATCH CONTEXT")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        with col_c1:
            st.metric("Wickets Remaining", f"{wickets_left}/10", help="Wickets in hand")
        with col_c2:
            overs_left = balls_left / 6
            st.metric("Overs Remaining", f"{overs_left:.1f}", help="Overs left to bat")
        with col_c3:
            run_rate_diff = rrr - crr
            st.metric("RR Difference", f"{run_rate_diff:.2f}", 
                     delta=f"{'Ahead' if run_rate_diff < 0 else 'Behind'}", 
                     help="Required RR - Current RR")
        with col_c4:
            runs_per_wicket = runs_left / wickets_left if wickets_left > 0 else runs_left
            st.metric("Runs per Wicket Needed", f"{runs_per_wicket:.1f}", 
                     help="Average runs needed per remaining wicket")

        # Get predictions from selected models
        predictions = {}
        
        if use_lr:
            lr_result = lr_pipe.predict_proba(input_df)
            predictions['Logistic Regression'] = {
                'batting': round(lr_result[0][1] * 100, 2),
                'bowling': round(lr_result[0][0] * 100, 2)
            }
        
        if use_rf:
            rf_result = rf_pipe.predict_proba(input_df)
            predictions['Random Forest'] = {
                'batting': round(rf_result[0][1] * 100, 2),
                'bowling': round(rf_result[0][0] * 100, 2)
            }

        # Display results
        st.markdown("### WIN PROBABILITY ANALYSIS")

        if show_comparison and len(predictions) > 1:
            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
            st.markdown("#### MODEL COMPARISON")
            
            for idx, (model_name, probs) in enumerate(predictions.items()):
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    st.markdown(f"""
                    <div class="probability-card-batting">
                        <div class="stat-label">Model: {model_name}</div>
                        <div style='font-size: 20px; color: white; margin: 10px 0;'>{batting_team}</div>
                        <div class="stat-value">{probs['batting']}%</div>
                        <div class="stat-label">Win Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown(f"""
                    <div class="probability-card-bowling">
                        <div class="stat-label">Model: {model_name}</div>
                        <div style='font-size: 20px; color: white; margin: 10px 0;'>{bowling_team}</div>
                        <div class="stat-value">{probs['bowling']}%</div>
                        <div class="stat-label">Win Probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if idx < len(predictions) - 1:
                    st.divider()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Average prediction
            if len(predictions) == 2:
                avg_batting = (predictions['Logistic Regression']['batting'] + predictions['Random Forest']['batting']) / 2
                avg_bowling = (predictions['Logistic Regression']['bowling'] + predictions['Random Forest']['bowling']) / 2
                
                st.markdown("#### ENSEMBLE PREDICTION")
                col_avg1, col_avg2 = st.columns(2)
                
                with col_avg1:
                    st.markdown(f"""
                    <div class="ensemble-box">
                        <div class="stat-label">Ensemble Average</div>
                        <div style='font-size: 22px; color: white; margin: 10px 0;'>{batting_team}</div>
                        <div class="stat-value">{avg_batting:.2f}%</div>
                        <div style='font-size: 14px; color: #B3E5FC; margin-top: 10px;'>Combined Model Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_avg2:
                    st.markdown(f"""
                    <div class="ensemble-box">
                        <div class="stat-label">Ensemble Average</div>
                        <div style='font-size: 22px; color: white; margin: 10px 0;'>{bowling_team}</div>
                        <div class="stat-value">{avg_bowling:.2f}%</div>
                        <div style='font-size: 14px; color: #B3E5FC; margin-top: 10px;'>Combined Model Prediction</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Show single model or first selected model
            for model_name, probs in predictions.items():
                col_result1, col_result2 = st.columns(2)

                with col_result1:
                    st.markdown(f"""
                    <div class="probability-card-batting">
                        <div class="stat-label">{model_name}</div>
                        <div style='font-size: 24px; color: white; margin: 10px 0;'>{batting_team}</div>
                        <div class="stat-value">{probs['batting']}%</div>
                        <div class="stat-label">Winning Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_result2:
                    st.markdown(f"""
                    <div class="probability-card-bowling">
                        <div class="stat-label">{model_name}</div>
                        <div style='font-size: 24px; color: white; margin: 10px 0;'>{bowling_team}</div>
                        <div class="stat-value">{probs['bowling']}%</div>
                        <div class="stat-label">Winning Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Detailed metrics table
        st.markdown("### DETAILED MATCH ANALYSIS")
        analysis_data = {
            'Metric': ['Target Score', 'Current Score', 'Runs Needed', 'Balls Remaining',
                      'Current Run Rate', 'Required Run Rate', 'Wickets Remaining', 
                      'Overs Remaining', 'Run Rate Difference', 'Runs per Wicket Needed'],
            'Value': [target, f"{score}/{wickets}", int(runs_left), int(balls_left),
                     f'{crr:.2f}', f'{rrr:.2f}', wickets_left, f'{balls_left/6:.1f}',
                     f'{rrr - crr:.2f}', f'{runs_left/wickets_left if wickets_left > 0 else runs_left:.1f}']
        }
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
        
        # Match situation analysis
        st.markdown("### MATCH SITUATION ANALYSIS")
        
        situation_text = ""
        situation_color = ""
        
        # Analyze match situation
        if rrr > 15:
            situation_text = "CRITICAL SITUATION - Very high required run rate. Batting team needs aggressive play."
            situation_color = "#d32f2f"
        elif rrr > 12:
            situation_text = "PRESSURE SITUATION - High required run rate. Batting team under pressure."
            situation_color = "#f57c00"
        elif rrr > 9:
            situation_text = "CHALLENGING SITUATION - Above average required run rate. Need good batting."
            situation_color = "#ffa000"
        elif rrr > 6:
            situation_text = "BALANCED SITUATION - Manageable required run rate. Match evenly poised."
            situation_color = "#0288d1"
        else:
            situation_text = "COMFORTABLE SITUATION - Low required run rate. Batting team in control."
            situation_color = "#388e3c"
        
        # Wickets situation
        if wickets_left <= 2:
            situation_text += f" Only {wickets_left} wickets remaining - tail-enders batting."
        elif wickets_left <= 4:
            situation_text += f" {wickets_left} wickets in hand - lower middle order exposed."
        elif wickets_left <= 6:
            situation_text += f" {wickets_left} wickets remaining - middle order batting."
        else:
            situation_text += f" {wickets_left} wickets in hand - top/middle order intact."
        
        st.markdown(f"""
        <div style='background: {situation_color}; color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h4 style='margin: 0; color: white;'>{situation_text}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Key insights
        st.markdown("### KEY INSIGHTS")
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.markdown("#### BATTING PERSPECTIVE")
            
            # Calculate what's needed
            boundaries_needed = runs_left / 4  # If all boundaries (4s)
            sixes_needed = runs_left / 6  # If all sixes
            
            st.markdown(f"""
            - Need **{boundaries_needed:.1f}** boundaries (all 4s)
            - Need **{sixes_needed:.1f}** sixes (all 6s)
            - Need **{runs_left/balls_left:.2f}** runs per ball
            - Avg **{runs_left/wickets_left if wickets_left > 0 else 0:.1f}** runs per wicket partnership needed
            """)
            
            # Batting strategy
            if rrr > 12:
                st.info("**Strategy:** Aggressive batting required. Look for boundaries every over.")
            elif rrr > 9:
                st.info("**Strategy:** Attack good balls. Build partnerships while scoring quickly.")
            else:
                st.info("**Strategy:** Play sensibly. Rotate strike and capitalize on loose deliveries.")
        
        with col_i2:
            st.markdown("#### BOWLING PERSPECTIVE")
            
            # Death overs check
            remaining_overs = balls_left / 6
            
            st.markdown(f"""
            - Can afford **{(balls_left * crr / 6):.1f}** runs at current rate
            - Must restrict to **{rrr:.2f}** runs per over
            - **{remaining_overs:.1f}** overs to defend
            - Need **{wickets_left}** wickets to win
            """)
            
            # Bowling strategy
            if wickets_left <= 3:
                st.success("**Strategy:** Target tail-enders. Defensive field. Dot balls crucial.")
            elif rrr > 12:
                st.success("**Strategy:** Bowling team ahead. Continue pressure. Restrict boundaries.")
            else:
                st.warning("**Strategy:** Need wickets. Attacking fields. Break partnerships.")
                
        # Prediction confidence
        st.markdown("### PREDICTION CONFIDENCE")
        
        # Calculate confidence based on model agreement
        if len(predictions) > 1:
            diff = abs(predictions['Logistic Regression']['batting'] - predictions['Random Forest']['batting'])
            if diff < 5:
                confidence = "Very High"
                conf_color = "#388e3c"
                conf_text = "Both models strongly agree on the prediction."
            elif diff < 10:
                confidence = "High"
                conf_color = "#0288d1"
                conf_text = "Models show good agreement on the prediction."
            elif diff < 20:
                confidence = "Medium"
                conf_color = "#ffa000"
                conf_text = "Models show moderate variation in predictions."
            else:
                confidence = "Low"
                conf_color = "#d32f2f"
                conf_text = "Significant variation between model predictions. Situation uncertain."
            
            st.markdown(f"""
            <div style='background: {conf_color}; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; margin: 0;'>Confidence Level: {confidence}</h4>
                <p style='color: white; margin: 10px 0 0 0;'>{conf_text}</p>
                <p style='color: #ffeb3b; margin: 5px 0 0 0; font-size: 14px;'>Model Variance: {diff:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Timestamp
        st.markdown(f"""
        <div style='text-align: center; color: #a8dadc; margin-top: 30px; padding: 15px;'>
            <p style='margin: 0;'>Prediction generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style='margin: 5px 0 0 0; font-size: 12px;'>Venue: {city} | Format: T20</p>
        </div>
        """, unsafe_allow_html=True)        
        st.markdown('<div class="success-message"><strong>Success:</strong> Prediction analysis completed successfully!</div>', unsafe_allow_html=True)
