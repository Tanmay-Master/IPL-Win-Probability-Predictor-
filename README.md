# IPL-Win-Probability-Predictor-
Developed an ML-powered IPL Win Probability Prediction System that provides real-time winning probabilities for cricket matches using machine learning algorithms and an interactive web interface.

1.	EXECUTIVE SUMMARY
Project Objective
Developed an ML-powered IPL Win Probability Prediction System that provides real-time winning probabilities for cricket matches using machine learning algorithms and an interactive web interface.
Key Achievements
•	Dual Model Architecture: Logistic Regression (79.47% accuracy) and Random Forest (99.36% accuracy)
•	Comprehensive Data Processing: Processed 1095 matches and 260,920 ball-by-ball records
•	Interactive Web Interface: User-friendly Streamlit application with real-time predictions
•	Production-Ready: Deployable system with model persistence and error handling



2.	PROJECT ARCHITECTURE
System Workflow
Data Collection → Data Preprocessing → Feature Engineering → Model Training → Web Application → Prediction
Component Architecture
Frontend (Streamlit)
    ↓
Model Layer (Pickle Files)
    ↓
Preprocessing Pipeline
    ↓
Data Layer (CSV Files)
3.	DATA COLLECTION & SOURCES
•	Data Files Description:
matches.csv - Match Metadata
Records: 1,095 matches
Columns: 20 features including:
•	Match ID, season, venue, teams
•	Toss details, winner, result margin
•	Player of match, umpires
deliveries.csv - Ball-by-Ball Data
Records: 260,920 deliveries
Columns: 17 features including:
•	Match ID, innings, teams
•	Over, ball, batsman, bowler
•	Runs, extras, wickets
•	Dismissal type, fielder
Initial Data Analysis
Matches Dataset:
•	Range: 0 to 1094 entries
•	Memory Usage: 171.2+ KB
•	Data Types: 
o	Numerical: 4 columns (int64: 1, float64: 3)
o	Categorical: 16 columns (object)
•	Deliveries Dataset:
o	Range: 0 to 260,919 entries
•	Data Types:
o	Numerical: 8 columns (int64)
o	Categorical: 9 columns (object)

4.	DATA PREPROCESSING PIPELINE
Step 1: Data Cleaning & Filtering
A.	Team Standardization
# Unified team names across datasets
'Delhi Daredevils' → 'Delhi Capitals'
'Deccan Chargers' → 'Sunrisers Hyderabad'
B.	Team Filtering
# Current IPL teams (10 franchises)
teams = [
    'Royal Challengers Bangalore', 'Mumbai Indians', 
    'Kolkata Knight Riders', 'Rajasthan Royals',
    'Chennai Super Kings', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Punjab Kings',
    'Lucknow Super Giants', 'Gujarat Titans'
]
C.	Match Filtering
•	Removed Duckworth-Lewis (D/L) method matches
•	Final Match Count**: 779 matches (from original 1095)


Step 2: Data Integration
A. Total Score Calculation
total_score_df = deliveries.groupby(["match_id", "inning"]).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
B. Dataset Merging
match_df = matches.merge(total_score_df[['match_id','total_runs']], 
                        left_on='id', right_on='match_id')
Step 3: Second Innings Focus
deliveries_df = match_df.merge(deliveries, on='match_id')
deliveries_df = deliveries_df[deliveries_df['inning'] == 2]
# Final deliveries count: 90,267 records

5.	FEATURE ENGINEERING
Key Features Created
1.	Match State Features
`runs_left`: `total_runs_x - current_score`
`balls_left`: `126 - (over * 6 + ball)`
`wickets_fallen`: Cumulative wickets fallen (10 - wickets remaining)
2.	Rate Calculations
`crr` (Current Run Rate): `(current_score * 6) / (120 - balls_left)`
`rrr` (Required Run Rate): `(runs_left * 6) / balls_left`
3.	Target Variable
`result`: Binary classification
`1` if batting team wins
`0` if bowling team wins


Data Quality Handling
# Handle infinite values
deliveries_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop NaN values
deliveries_df.dropna(subset=['crr', 'rrr'], inplace=True)
# Final dataset shape after cleaning
final_df.shape  # Sampled and cleaned dataset
6.	MACHINE LEARNING MODELS
Model 1: Logistic Regression
Configuration
Pipeline([
    ('preprocessor', ColumnTransformer),
    ('model', LogisticRegression(solver='liblinear'))
])
Characteristics
•	Type: Linear classification model
•	Advantages: Fast, interpretable, less prone to overfitting
•	Use Case: Baseline model and quick predictions
Model 2: Random Forest Classifier
Configuration
Pipeline([
    ('preprocessor', ColumnTransformer),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
Characteristics
•	Type: Ensemble learning (Decision Trees)
•	Advantages: High accuracy, handles non-linear relationships
•	Risk: Potential overfitting
Preprocessing Pipeline
Column Transformer
trf = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols),
    ('num', SimpleImputer(strategy='mean'), num_cols)
], remainder='passthrough')
Categorical Features
cat_cols = ['batting_team', 'bowling_team', 'city']

7.	MODEL TRAINING & EVALUATION
Training Setup
•	Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
•	Final Features Used
X = final_df[[
    'batting_team', 'bowling_team', 'city', 
    'runs_left', 'balls_left', 'wickets_fallen',
    'total_runs_x', 'crr', 'rrr'
]]
y = final_df['result']
•	Performance Metrics
Accuracy Scores:
Model	Accuracy	Performance
Logistic Regression	79.47%	Good baseline
Random Forest	99.36%	Excellent (potential overfitting)

Model Persistence
# Save trained models
pickle.dump(lr_pipe, open('lr_pipe.pkl', 'wb'))
pickle.dump(rf_pipe, open('rf_pipe.pkl', 'wb'))

8.	WEB APPLICATION DEVELOPMENT
Streamlit Frontend Architecture
A. Page Configuration
st.set_page_config(
    page_title="IPL Win Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
B. Custom Styling System
•	Gradient Backgrounds: Dark blue theme matching IPL
•	Animated Components: Hover effects and transitions
•	Color-coded Cards: Green (batting) and Orange (bowling)
•	Responsive Design: Mobile-friendly layout
Application Components:
1. Header Section
•	Main Title: "IPL WIN PROBABILITY PREDICTOR"
•	Subtitle: "Advanced Machine Learning Analysis"
•	Visual Design: Gradient background with shadow effects
2. Sidebar Features
•	About Section: Project description
•	Model Information: LR and RF characteristics
•	Key Metrics: Explanation of used parameters
3. Main Input Section
Team Selection
# 10 IPL Teams
teams = ['Royal Challengers Bangalore', 'Mumbai Indians', ...]
Venue Selection
# 32 International Cities
cities = ['Bangalore', 'Delhi', 'Mumbai', 'Kolkata', ...]
Match Parameters
•	Target Score
•	Current Score
•	Overs Completed
•	Wickets Lost
4. Model Selection Interface
•	Checkbox System: Select LR, RF, or both
•	Comparison Mode: Side-by-side model results
•	Ensemble Prediction: Average of both models
UI/UX Features
Visual Components
1. Metric Cards: 
•	Current Run Rate (CRR)
•	Required Run Rate (RRR)
•	Runs Needed
•	Balls Remaining
2. Probability Cards:
•	Batting Team (Green gradient)
•	Bowling Team (Orange gradient)
•	Large percentage display
3. Analysis Table:
•	Comprehensive match metrics
•	Real-time calculations
Interactive Elements
•	Hover Effects: Button animations
•	Validation Checks: Team selection validation
•	Error Handling: Invalid input prevention
•	Success Messages: Operation confirmation

9.	TECHNICAL IMPLEMENTATION DETAILS
Backend Technical Stack
•	Libraries Used
# Data Processing
pandas, numpy
•	Visualization
o	seaborn, matplotlib
•	Machine Learning
o	scikit-learn
•	Model Persistence
o	pickle
•	Web Framework
o	streamlit
Data Processing Pipeline
1.	Data Loading: CSV files with error handling
2.	Data Cleaning: Null value treatment, team standardization
3.	Feature Engineering: Real-time match state calculations
4.	Model Integration: Pre-trained model loading
Frontend Technical Stack
Streamlit Components:
•	Input Widgets: selectbox, number_input, checkbox, button
•	Layout Management: columns, containers, expanders
•	Data Display: dataframe, metrics, JSON

Custom CSS Features
/* Gradient Backgrounds */
background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
/* Animated Buttons */
transition: all 0.3s ease;
box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
/* Card Designs */
border-radius: 12px;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
Match Metrics Calculator
def calculate_metrics(score, overs, wickets, target):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    return runs_left, balls_left, wickets_left, crr, rrr
Prediction Engine
def generate_prediction(input_df, model):
    probabilities = model.predict_proba(input_df)
    batting_prob = round(probabilities[0][1] * 100, 2)
    bowling_prob = round(probabilities[0][0] * 100, 2)
    return batting_prob, bowling_prob

10.	PERFORMANCE ANALYSIS
Model Performance Deep Dive
Logistic Regression Analysis:
•	Accuracy: 79.47%
•	Strengths: 
	Fast inference time
	Model interpretability
	Less memory consumption
•	Limitations:
	Assumes linear relationships
	Lower accuracy compared to ensemble methods
Random Forest Analysis
•	-Accuracy: 99.36%
•	Strengths:
	Handles non-linear patterns
	Robust to outliers
	Feature importance analysis
•	Concerns:
	Potential overfitting
	Higher computational cost
	Black box nature
Business Logic Implementation
Input Validation System
# Team validation
if batting_team == bowling_team:
    show_error("Batting and Bowling teams must be different!")
# Match progress validation
if score > target:
    show_success("Match Already Won!")
elif overs == 0:
    show_warning("Overs completed must be greater than 0!")

Real-time Feature Calculation
•	Dynamic CRR and RRR updates
•	Live wicket count tracking
•	Progressive match state analysis

11.	DEPLOYMENT & USAGE
Deployment Ready Features
Model Loading System
@st.cache_resource
def load_models():
    lr_pipe = pickle.load(open('lr_pipe.pkl', 'rb'))
    rf_pipe = pickle.load(open('rf_pipe.pkl', 'rb'))
    return lr_pipe, rf_pipe
Production Considerations
•	Caching: Efficient model loading
•	Error Handling: Graceful failure management
•	Scalability: Support for multiple concurrent users
User Guide
Step-by-Step Usage
1. Model Selection: Choose prediction algorithms
2. Team Setup: Select competing teams and venue
3. Match Input: Enter current match situation
4. Generate: Get instant probability analysis
5. Compare: View different model outputs
Use Cases
•	Live Match Analysis: Real-time probability updates
•	Strategy Planning: Team selection and batting order
•	Educational Tool: Understanding match dynamics
•	Broadcast Enhancement: TV commentary support

12.	FUTURE ENHANCEMENTS
Planned Improvements
A. Model Enhancements
1. Additional Algorithms:
•	XGBoost, Neural Networks
•	Time-series models for progressive analysis
2. Feature Expansion:
•	Player-specific statistics
•	Pitch condition analysis
•	Weather impact modeling
B. Application Features
1. Real-time Data Integration:
•	Live match data APIs
•	Automatic score updates
•	Historical performance tracking
2. Advanced Visualization:
•	Probability progression charts
•	Win probability over time
•	Interactive match simulations
3. Mobile Application:
•	Native iOS/Android apps
•	Push notifications
•	Offline functionality
C. Technical Improvements
1. Performance Optimization:
•	Database integration
•	Caching mechanisms
•	Async processing
2. Scalability Features:
•	Multi-user support
•	Cloud deployment
•	API endpoints
Long-term Vision
Transform into a comprehensive cricket analytics platform with:
•	Player Performance Predictions
•	Team Strategy Recommendations
•	Tournament Simulation Capabilities
•	Broadcast Integration Tools

Conclusion
This IPL Win Probability Predictor represents a successful integration of machine learning and web technologies to solve real-world sports analytics challenges. The system demonstrates:

•	Robust Data Processing- Handled large-scale cricket data  
•	Advanced ML Implementation- Multiple model approaches  
•	User-Centric Design- Intuitive and engaging interface  
•	Production Readiness - Deployable and scalable architecture  
•	Educational Value- Makes complex analytics accessible  
The project serves as a foundation for sports analytics applications and showcases the potential of AI in en
