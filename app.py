"""
╔══════════════════════════════════════════════════════════════╗
║   STUDENT SKILL RECOMMENDATION SYSTEM - SIMPLIFIED          ║
║   Easy to understand charts and features                    ║
║   Run: streamlit run app.py                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Page Config
st.set_page_config(
    page_title="SkillRec AI | Student Recommendation",
    page_icon="🎓", 
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header {
        background: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SKILL_COLS = ["Skill1", "Skill2", "Skill3", "Skill4"]
SUBJECT_COLS = ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"]

# Names for better display
SKILL_NAMES = {
    "Skill1": "🧠 Logical Reasoning",
    "Skill2": "🔍 Analytical Thinking", 
    "Skill3": "💡 Problem Solving",
    "Skill4": "🎨 Creative Thinking"
}

SUBJECT_NAMES = {
    "Subject1": "📐 Mathematics",
    "Subject2": "🔬 Science",
    "Subject3": "💻 Technology",
    "Subject4": "🎭 Arts",
    "Subject5": "📊 Business"
}

BASE = os.path.dirname(os.path.abspath(__file__))

# Generate sample data if files don't exist
def generate_sample_data():
    """Generate simple sample data with correct column names"""
    
    # Expert Skill Map
    expert_path = os.path.join(BASE, "ExpertSkillMap.csv")
    if not os.path.exists(expert_path):
        expert_data = {
            "Subject": ["Mathematics", "Science", "Technology", "Arts", "Business"],
            "Skill1": [1, 2, 3, 4, 2],
            "Skill2": [2, 1, 4, 3, 3],
            "Skill3": [3, 4, 1, 2, 4],
            "Skill4": [4, 3, 2, 1, 1]
        }
        pd.DataFrame(expert_data).to_csv(expert_path, index=False)
        print("Created ExpertSkillMap.csv")
    
    # Student Cognitive Skills
    cog_path = os.path.join(BASE, "StudentCognitiveSkill.csv")
    if not os.path.exists(cog_path):
        np.random.seed(42)
        cog_data = {
            "Student": [f"S{i}" for i in range(1, 501)],
            "Skill1": np.random.randint(1, 5, 500),
            "Skill2": np.random.randint(1, 5, 500),
            "Skill3": np.random.randint(1, 5, 500),
            "Skill4": np.random.randint(1, 5, 500)
        }
        pd.DataFrame(cog_data).to_csv(cog_path, index=False)
        print("Created StudentCognitiveSkill.csv")
    
    # Student Subject Knowledge
    sub_path = os.path.join(BASE, "StudentSubjectKnowledge.csv")
    if not os.path.exists(sub_path):
        np.random.seed(42)
        sub_data = {
            "Student": [f"S{i}" for i in range(1, 501)],
            "Subject1": np.random.randint(1, 6, 500),
            "Subject2": np.random.randint(1, 6, 500),
            "Subject3": np.random.randint(1, 6, 500),
            "Subject4": np.random.randint(1, 6, 500),
            "Subject5": np.random.randint(1, 6, 500)
        }
        pd.DataFrame(sub_data).to_csv(sub_path, index=False)
        print("Created StudentSubjectKnowledge.csv")

# Load data with correct column handling
@st.cache_data
def load_data():
    generate_sample_data()
    
    # Load files
    expert = pd.read_csv(os.path.join(BASE, "ExpertSkillMap.csv"))
    cognitive = pd.read_csv(os.path.join(BASE, "StudentCognitiveSkill.csv"))
    subject = pd.read_csv(os.path.join(BASE, "StudentSubjectKnowledge.csv"))
    
    # Ensure column names are consistent
    # Check if columns are named differently and rename if needed
    if 'Subject1' not in subject.columns:
        # If columns are named differently, rename them
        subject_cols = subject.columns.tolist()
        for i, col in enumerate(subject_cols):
            if col != 'Student' and i <= 5:
                subject.rename(columns={col: f'Subject{i}'}, inplace=True)
    
    return expert, cognitive, subject

# Simple feature engineering - easy to understand
def create_features(df):
    """Create simple, understandable features"""
    df = df.copy()
    
    # Check which columns exist
    skill_cols_present = [col for col in SKILL_COLS if col in df.columns]
    subject_cols_present = [col for col in SUBJECT_COLS if col in df.columns]
    
    # Basic statistics (easy to understand)
    if len(skill_cols_present) > 0:
        df['avg_skill'] = df[skill_cols_present].mean(axis=1).round(2)  # Average of all skills
        df['total_skill'] = df[skill_cols_present].sum(axis=1)  # Sum of all skills
        df['strongest_skill'] = df[skill_cols_present].idxmax(axis=1)  # Best skill
        df['weakest_skill'] = df[skill_cols_present].idxmin(axis=1)  # Worst skill
        df['skill_range'] = df[skill_cols_present].max(axis=1) - df[skill_cols_present].min(axis=1)  # Skill range
    
    if len(subject_cols_present) > 0:
        df['avg_subject'] = df[subject_cols_present].mean(axis=1).round(2)  # Average subject knowledge
        df['total_subject'] = df[subject_cols_present].sum(axis=1)  # Total subject knowledge
        df['best_subject'] = df[subject_cols_present].idxmax(axis=1)  # Best subject
        df['worst_subject'] = df[subject_cols_present].idxmin(axis=1)  # Worst subject
    
    # Overall score (combines skills and knowledge)
    if 'avg_skill' in df.columns and 'avg_subject' in df.columns:
        df['overall_score'] = (df['avg_skill'] * 0.6 + df['avg_subject'] * 0.4).round(2)
    elif 'avg_skill' in df.columns:
        df['overall_score'] = df['avg_skill'].round(2)
    elif 'avg_subject' in df.columns:
        df['overall_score'] = df['avg_subject'].round(2)
    
    return df

# Build target (which subject to recommend)
def build_target(df, expert):
    """Simple target building using expert mapping"""
    targets = []
    
    # Get skill columns that exist
    skill_cols = [col for col in SKILL_COLS if col in df.columns]
    
    for idx, row in df.iterrows():
        student_skills = np.array([row[col] for col in skill_cols])
        
        best_match = None
        best_score = -1
        
        for _, exp_row in expert.iterrows():
            exp_skills = np.array([exp_row[col] for col in skill_cols])
            # Simple similarity score (lower is better)
            score = np.sum(np.abs(student_skills - exp_skills))
            
            if best_match is None or score < best_score:
                best_score = score
                best_match = exp_row['Subject']
        
        targets.append(best_match)
    
    return targets

# Train models
@st.cache_resource
def train_models():
    expert, cognitive, subject = load_data()
    
    # Merge data
    df = pd.merge(cognitive, subject, on='Student')
    
    # Create target
    df['Target'] = build_target(df, expert)
    
    # Create features
    df = create_features(df)
    
    # Get available columns for features
    available_skill_cols = [col for col in SKILL_COLS if col in df.columns]
    available_subject_cols = [col for col in SUBJECT_COLS if col in df.columns]
    
    # Select features for training
    feature_cols = (available_skill_cols + available_subject_cols + 
                   [col for col in ['avg_skill', 'total_skill', 'avg_subject', 
                                   'total_subject', 'skill_range', 'overall_score'] 
                    if col in df.columns])
    
    X = df[feature_cols]
    y = df['Target']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train and evaluate
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = name
    
    # Save best model
    bundle = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feature_cols,
        'classes': le.classes_
    }
    
    return bundle, df, results

# Simple EDA functions
def plot_skill_distribution(df):
    """Simple bar chart for skill distribution"""
    available_skills = [col for col in SKILL_COLS if col in df.columns]
    if not available_skills:
        return None
    
    n_skills = len(available_skills)
    n_rows = (n_skills + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, skill in enumerate(available_skills):
        skill_counts = df[skill].value_counts().sort_index()
        axes[i].bar(skill_counts.index, skill_counts.values, color='#667eea', edgecolor='black')
        axes[i].set_title(SKILL_NAMES.get(skill, skill), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Rating (1-4)', fontsize=10)
        axes[i].set_ylabel('Number of Students', fontsize=10)
        axes[i].grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_skills), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_subject_distribution(df):
    """Simple bar chart for subject distribution"""
    available_subjects = [col for col in SUBJECT_COLS if col in df.columns]
    if not available_subjects:
        return None
    
    n_subjects = len(available_subjects)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, subject in enumerate(available_subjects):
        subject_counts = df[subject].value_counts().sort_index()
        axes[i].bar(subject_counts.index, subject_counts.values, color='#764ba2', edgecolor='black')
        axes[i].set_title(SUBJECT_NAMES.get(subject, subject), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Rating (1-5)', fontsize=10)
        axes[i].set_ylabel('Number of Students', fontsize=10)
        axes[i].grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(available_subjects), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_target_distribution(df):
    """Simple pie chart for target distribution"""
    if 'Target' not in df.columns:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    target_counts = df['Target'].value_counts()
    
    # Bar chart
    colors = plt.cm.Set3(range(len(target_counts)))
    bars = ax1.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black')
    ax1.set_title('Number of Students per Subject', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Recommended Subject', fontsize=10)
    ax1.set_ylabel('Number of Students', fontsize=10)
    ax1.set_xticklabels(target_counts.index, rotation=45, ha='right')
    
    # Add values on bars
    for bar, val in zip(bars, target_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(val), ha='center', fontweight='bold')
    
    # Pie chart
    ax2.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax2.set_title('Subject Distribution (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_skill_by_target(df):
    """Simple boxplot showing skills by recommended subject"""
    if 'Target' not in df.columns:
        return None
    
    available_skills = [col for col in SKILL_COLS if col in df.columns]
    if not available_skills:
        return None
    
    n_skills = len(available_skills)
    n_rows = (n_skills + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    targets = df['Target'].unique()
    
    for i, skill in enumerate(available_skills):
        data_to_plot = [df[df['Target'] == target][skill].values for target in targets]
        bp = axes[i].boxplot(data_to_plot, labels=targets, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set3(range(len(data_to_plot)))):
            patch.set_facecolor(color)
        
        axes[i].set_title(SKILL_NAMES.get(skill, skill), fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Recommended Subject', fontsize=10)
        axes[i].set_ylabel('Skill Rating (1-4)', fontsize=10)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(available_skills), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🎓 Student Skill-Based Course Recommendation System</h1>
        <p>Find the best subject match based on your skills and knowledge</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train models
    with st.spinner('Loading system...'):
        try:
            bundle, df, results = train_models()
        except Exception as e:
            st.error(f"Error loading system: {str(e)}")
            st.info("Please make sure the CSV files have the correct format.")
            return
    
    # Sidebar navigation
    st.sidebar.markdown("## 📚 Navigation")
    page = st.sidebar.radio("Go to", 
        ["📖 Problem Statement", 
         "📊 Dataset Overview", 
         "🔍 Simple EDA", 
         "⚙️ Features Explained", 
         "🤖 Model Results",
         "🎯 Make Prediction"])
    
    # Show best model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 🏆 Best Model")
    st.sidebar.markdown(f"**{bundle['model_name']}**")
    st.sidebar.markdown(f"Accuracy: **{results[bundle['model_name']]['accuracy']*100:.1f}%**")
    
    # Page 1: Problem Statement
    if page == "📖 Problem Statement":
        st.markdown("## 🎯 Problem Statement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### The Challenge
            Students often struggle to choose the right subjects because:
            - They don't know their strengths
            - They follow friends instead of their interests
            - They lack guidance on subject requirements
            
            ### Our Solution
            We use Machine Learning to analyze:
            - **4 Cognitive Skills** (Logical, Analytical, Problem Solving, Creative)
            - **5 Subject Knowledge Areas** (Math, Science, Tech, Arts, Business)
            
            Then recommend the best subject match!
            """)
        
        with col2:
            st.markdown("""
            ### How It Works
            1. **Input** your skill ratings (1-4) and subject knowledge (1-5)
            2. **System analyzes** your strengths and patterns
            3. **ML Model** predicts the best subject for you
            4. **Get recommendation** with confidence score
            
            ### Benefits
            ✅ Personalized recommendations
            ✅ Based on actual data, not guesswork
            ✅ Helps in career planning
            """)
            
            st.info("💡 **Tip**: Higher ratings mean better skills/knowledge!")
    
    # Page 2: Dataset Overview
    elif page == "📊 Dataset Overview":
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Students", len(df), "500+")
        with col2:
            skill_cols = [col for col in SKILL_COLS if col in df.columns]
            st.metric("Skills Measured", len(skill_cols), "1-4 scale")
        with col3:
            subject_cols = [col for col in SUBJECT_COLS if col in df.columns]
            st.metric("Subjects", len(subject_cols), "1-5 scale")
        
        st.markdown("### Sample Data")
        tab1, tab2, tab3 = st.tabs(["Skills Data", "Subjects Data", "Target Distribution"])
        
        # Get available columns
        available_skills = [col for col in SKILL_COLS if col in df.columns]
        available_subjects = [col for col in SUBJECT_COLS if col in df.columns]
        
        with tab1:
            if available_skills:
                st.dataframe(df[['Student'] + available_skills].head(10), use_container_width=True)
                st.caption("Skill ratings: 1=Lowest, 4=Highest")
            else:
                st.warning("No skill data available")
        
        with tab2:
            if available_subjects:
                st.dataframe(df[['Student'] + available_subjects].head(10), use_container_width=True)
                st.caption("Subject knowledge: 1=Beginner, 5=Expert")
            else:
                st.warning("No subject data available")
        
        with tab3:
            if 'Target' in df.columns:
                target_counts = df['Target'].value_counts()
                st.dataframe(target_counts.reset_index().rename(columns={'index': 'Subject', 'Target': 'Count'}), 
                            use_container_width=True)
            else:
                st.warning("No target data available")
    
    # Page 3: Simple EDA
    elif page == "🔍 Simple EDA":
        st.markdown("## 🔍 Exploratory Data Analysis")
        st.markdown("Simple charts to understand our data")
        
        # Skill distributions
        st.markdown("### 🧠 Skills Distribution")
        fig = plot_skill_distribution(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No skill data available for visualization")
        
        # Subject distributions
        st.markdown("### 📚 Subject Knowledge Distribution")
        fig = plot_subject_distribution(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No subject data available for visualization")
        
        # Target distribution
        st.markdown("### 🎯 Recommended Subjects Distribution")
        fig = plot_target_distribution(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No target data available for visualization")
        
        # Skills by target
        st.markdown("### 📊 Skills by Recommended Subject")
        fig = plot_skill_by_target(df)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Insufficient data for skills by target visualization")
    
    # Page 4: Features Explained
    elif page == "⚙️ Features Explained":
        st.markdown("## ⚙️ Simple Features We Created")
        
        st.markdown("""
        ### Why Create New Features?
        Raw data (just skills and subjects) is good, but creating new features helps the model find patterns better!
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Basic Statistics
            - **Average Skill**: Your overall skill level
            - **Total Skill**: Sum of all skills
            - **Strongest Skill**: Your best skill
            - **Weakest Skill**: Area for improvement
            """)
            
            st.markdown("""
            ### 📚 Subject Statistics
            - **Average Subject**: Overall knowledge
            - **Total Subject**: Knowledge sum
            - **Best Subject**: Your strongest area
            - **Worst Subject**: Area to improve
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Advanced Features
            - **Skill Range**: How specialized vs balanced you are
              - Small range = Balanced skills
              - Large range = Specialized in one area
            
            - **Overall Score**: Combines skills (60%) and knowledge (40%)
              - Higher score = Better prepared for any subject
            """)
        
        # Show example of features
        feature_cols = ['Student']
        for col in ['avg_skill', 'strongest_skill', 'avg_subject', 'best_subject', 'overall_score', 'Target']:
            if col in df.columns:
                feature_cols.append(col)
        
        if len(feature_cols) > 1:
            st.markdown("### Example: Student Profile with Features")
            sample_df = df[feature_cols].head(5)
            st.dataframe(sample_df, use_container_width=True)
        
        st.info("💡 These features help the model understand each student's unique profile!")
    
    # Page 5: Model Results
    elif page == "🤖 Model Results":
        st.markdown("## 🤖 Model Performance")
        
        # Results table
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': f"{res['accuracy']*100:.1f}%",
                'F1 Score': f"{res['f1_score']*100:.1f}%"
            }
            for name, res in results.items()
        ]).sort_values('Accuracy', ascending=False)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best model
        best = results[bundle['model_name']]
        st.success(f"🏆 **Best Model: {bundle['model_name']}** with {best['accuracy']*100:.1f}% accuracy!")
        
        # Feature importance (only for tree-based models)
        if hasattr(bundle['model'], 'feature_importances_') and len(bundle['feature_cols']) > 0:
            st.markdown("### 🔑 Feature Importance")
            st.markdown("Which features matter most for prediction?")
            
            importances = bundle['model'].feature_importances_
            feature_names = bundle['feature_cols']
            
            # Create readable feature names
            readable_names = []
            for f in feature_names:
                if f in SKILL_NAMES:
                    readable_names.append(SKILL_NAMES[f])
                elif f in SUBJECT_NAMES:
                    readable_names.append(SUBJECT_NAMES[f])
                else:
                    readable_names.append(f.replace('_', ' ').title())
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            n_features = min(10, len(importances))
            indices = np.argsort(importances)[-n_features:]  # Top n
            
            ax.barh(range(len(indices)), importances[indices], color='#667eea')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([readable_names[i] for i in indices])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {n_features} Most Important Features', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("### 📈 What These Metrics Mean")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Accuracy**: Percentage of correct predictions
            - 100% = Perfect predictions
            - 50% = Random guessing
            - Higher is better!
            """)
        
        with col2:
            st.markdown("""
            **F1 Score**: Balance of precision and recall
            - Considers both false positives and false negatives
            - Good measure for balanced evaluation
            - Higher is better!
            """)
    
    # Page 6: Make Prediction
    else:
        st.markdown("## 🎯 Get Your Subject Recommendation")
        
        st.markdown("### Enter Your Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Cognitive Skills (1-4)")
            skill1 = st.slider("Logical Reasoning", 1, 4, 2, help="Ability to think logically and solve puzzles")
            skill2 = st.slider("Analytical Thinking", 1, 4, 2, help="Ability to analyze data and find patterns")
            skill3 = st.slider("Problem Solving", 1, 4, 2, help="Ability to solve complex problems")
            skill4 = st.slider("Creative Thinking", 1, 4, 2, help="Ability to think creatively and innovate")
        
        with col2:
            st.markdown("#### 📚 Subject Knowledge (1-5)")
            subj1 = st.slider("Mathematics", 1, 5, 3, help="Knowledge of math and logic")
            subj2 = st.slider("Science", 1, 5, 3, help="Knowledge of science and research")
            subj3 = st.slider("Technology", 1, 5, 3, help="Knowledge of tech and computing")
            subj4 = st.slider("Arts", 1, 5, 3, help="Knowledge of arts and humanities")
            subj5 = st.slider("Business", 1, 5, 3, help="Knowledge of business and social studies")
        
        if st.button("🔮 Get Recommendation", type="primary"):
            try:
                # Create feature vector
                features = np.array([
                    skill1, skill2, skill3, skill4,  # Skills
                    subj1, subj2, subj3, subj4, subj5,  # Subjects
                    (skill1+skill2+skill3+skill4)/4,  # avg_skill
                    skill1+skill2+skill3+skill4,  # total_skill
                    (subj1+subj2+subj3+subj4+subj5)/5,  # avg_subject
                    subj1+subj2+subj3+subj4+subj5,  # total_subject
                    max(skill1,skill2,skill3,skill4) - min(skill1,skill2,skill3,skill4),  # skill_range
                    ((skill1+skill2+skill3+skill4)/4)*0.6 + ((subj1+subj2+subj3+subj4+subj5)/5)*0.4  # overall_score
                ]).reshape(1, -1)
                
                # Scale features
                features_scaled = bundle['scaler'].transform(features)
                
                # Predict
                prediction = bundle['model'].predict(features_scaled)[0]
                probabilities = bundle['model'].predict_proba(features_scaled)[0]
                
                # Get predicted subject
                predicted_subject = bundle['label_encoder'].inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100
                
                # Display results
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>🎉 Recommended Subject</h2>
                    <h1 style='font-size: 3rem; margin: 0.5rem 0;'>{predicted_subject}</h1>
                    <p style='font-size: 1.2rem;'>{SUBJECT_NAMES.get(predicted_subject, predicted_subject)}</p>
                    <p style='font-size: 1.5rem; margin-top: 1rem;'>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show all probabilities
                st.markdown("### 📊 All Subject Probabilities")
                prob_df = pd.DataFrame({
                    'Subject': bundle['label_encoder'].classes_,
                    'Probability (%)': probabilities * 100
                }).sort_values('Probability (%)', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#667eea' if s != predicted_subject else '#ff6b6b' for s in prob_df['Subject']]
                ax.barh(prob_df['Subject'], prob_df['Probability (%)'], color=colors, edgecolor='black')
                ax.set_xlabel('Probability (%)')
                ax.set_title('Subject Recommendation Probabilities')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show strengths
                st.markdown("### 💪 Your Strengths")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Skills:**")
                    skills_dict = {
                        'Logical Reasoning': skill1,
                        'Analytical Thinking': skill2,
                        'Problem Solving': skill3,
                        'Creative Thinking': skill4
                    }
                    top_skills = sorted(skills_dict.items(), key=lambda x: x[1], reverse=True)[:2]
                    for skill, rating in top_skills:
                        st.markdown(f"- {skill}: {rating}/4 ⭐")
                
                with col2:
                    st.markdown("**Top Subjects:**")
                    subjects_dict = {
                        'Mathematics': subj1,
                        'Science': subj2,
                        'Technology': subj3,
                        'Arts': subj4,
                        'Business': subj5
                    }
                    top_subjects = sorted(subjects_dict.items(), key=lambda x: x[1], reverse=True)[:2]
                    for subject, rating in top_subjects:
                        st.markdown(f"- {subject}: {rating}/5 ⭐")
                
                st.info(f"💡 **Why {predicted_subject}?** Based on your skills and knowledge, you're best suited for {predicted_subject}. Your strongest skill is {top_skills[0][0]} ({top_skills[0][1]}/4) and strongest subject is {top_subjects[0][0]} ({top_subjects[0][1]}/5).")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check if the model is properly trained.")

if __name__ == "__main__":
    main()