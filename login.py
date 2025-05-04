import streamlit as st
import time
import pandas as pd
# import plotly.express as px # No longer needed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Configure page
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# --- CSS Definitions ---

# Custom CSS for Login Page (Includes background and bubbles)
def login_css():
    st.markdown("""
    <style>
        /* Main container */
        .stApp {
            background: linear-gradient(45deg, #f3f4f6 0%, #fff 100%);
            height: 100vh !important;
            overflow: hidden !important;
            position: relative;
        }
        
        /* Additional scroll prevention */
        [data-testid="stAppViewContainer"], 
        .main,
        .block-container,
        section[data-testid="stSidebar"] {
            height: 100vh !important;
            overflow: hidden !important;
        }
        
        /* Bubbles */
        .bubbles {
            position: fixed;
            width: 100%;
            height: 100vh;
            z-index: 0;
            overflow: hidden;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        
        .bubble {
            position: absolute;
            bottom: -100px;
            width: 40px;
            height: 40px;
            background: linear-gradient(45deg, #3b82f6 10%, rgba(59, 130, 246, 0.3));
            border-radius: 50%;
            opacity: 0.5;
            animation: rise 8s infinite ease-in;
        }
        
        .bubble:nth-child(1) { width: 40px; height: 40px; left: 10%; animation-duration: 8s; }
        .bubble:nth-child(2) { width: 20px; height: 20px; left: 20%; animation-duration: 5s; animation-delay: 1s; }
        .bubble:nth-child(3) { width: 50px; height: 50px; left: 35%; animation-duration: 7s; animation-delay: 2s; }
        .bubble:nth-child(4) { width: 30px; height: 30px; left: 50%; animation-duration: 6s; animation-delay: 0s; }
        .bubble:nth-child(5) { width: 35px; height: 35px; left: 65%; animation-duration: 9s; animation-delay: 1.5s; }
        .bubble:nth-child(6) { width: 45px; height: 45px; left: 80%; animation-duration: 8s; animation-delay: 3s; }
        .bubble:nth-child(7) { width: 25px; height: 25px; left: 90%; animation-duration: 7s; animation-delay: 2s; }
        
        @keyframes rise {
            0% { bottom: -100px; transform: translateX(0); }
            50% { transform: translateX(100px); }
            100% { bottom: 100vh; transform: translateX(-100px); }
        }
        
        /* Hide Streamlit elements needed for login */
        #MainMenu, header, footer, .stDeployButton {
            display: none !important;
        }
        section[data-testid="stSidebar"] { /* Hide sidebar on login */
             display: none !important;
        }
        
        /* Login container */
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 1;
            padding: 0; margin: 0;
            overflow: hidden;
        }
        
        /* Login box */
        .login-box {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 40px 30px !important;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
            position: relative;
            z-index: 2;
        }
        
        /* Title */
        .login-title { text-align: center; font-size: 24px; color: #1a202c; margin-bottom: 50px !important; font-weight: 600; letter-spacing: -0.5px; }
        
        /* Input fields container */
        .stTextInput > div { display: flex !important; justify-content: center !important; width: 100% !important; max-width: 300px !important; margin: 0 auto !important; position: relative !important; z-index: 10 !important; }
        /* Input field wrapper */
        .stTextInput > div > div { width: 100% !important; max-width: 300px !important; margin: 0 auto !important; padding: 0 !important; position: relative !important; z-index: 10 !important; }
        /* Input fields */
        .stTextInput > div > div > input {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 0 16px !important;
            font-size: 15px !important;
            color: #1a202c !important;
            height: 45px !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
            margin-bottom: 25px !important;
            box-sizing: border-box !important;
            position: relative !important;
            z-index: 10 !important;
        }
        .stTextInput > div > div > input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important; outline: none !important; }
        
        /* Login button */
        .stButton { display: flex !important; justify-content: center !important; margin-top: 20px !important; position: relative !important; z-index: 100 !important; }
        .stButton > button { background: #3b82f6 !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 12px 24px !important; font-size: 15px !important; font-weight: 500 !important; width: 90px !important; height: 40px !important; margin-top: 10px !important; transition: background 0.3s ease !important; position: relative !important; z-index: 100 !important; cursor: pointer !important; }
        .stButton > button:hover { background: #2563eb !important; }
        
        /* Error Messages */
        .error-message { background: #fed7d7; color: #c53030; padding: 8px; border-radius: 4px; text-align: center; margin-top: 10px; font-size: 14px; }
        
        /* Password visibility toggle */
        button[aria-label="Toggle password visibility"] { height: 45px !important; margin-top: 0 !important; color: #718096 !important; background: transparent !important; border: none !important; padding: 0 12px !important; position: relative !important; z-index: 10 !important; }
        
    </style>

    <div class="bubbles">
        <div class="bubble"></div> <div class="bubble"></div> <div class="bubble"></div>
        <div class="bubble"></div> <div class="bubble"></div> <div class="bubble"></div>
        <div class="bubble"></div>
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for Main Dashboard (New Design based on image)
def main_dashboard_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        /* Apply background gradient and font */
        .stApp {
            background: linear-gradient(135deg, #00B4DB 0%, #0083B0 40%, #4a00e0 100%); /* Blue to purple gradient */
            height: 100vh !important;
            overflow: auto !important; /* Allow scroll for content */
            font-family: 'Poppins', sans-serif;
        }

        /* Hide elements not needed in this design */
        #MainMenu, header, footer, .stDeployButton, section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* Center content */
        .block-container {
             padding: 2rem 1rem 1rem 1rem !important;
             max-width: 800px !important; /* Adjust max-width as needed */
             margin: 2rem auto !important; /* Center content vertically and horizontally */
             display: flex;
             flex-direction: column;
             align-items: center; /* Center items horizontally */
        }

        /* Titles */
        h1, h2, h3 {
            color: white;
            text-align: center;
        }
        h1 {
            font-weight: 600;
            margin-bottom: 1rem;
        }
        h3 {
             font-weight: 500;
             margin-bottom: 0.5rem;
             margin-top: 1.5rem;
             align-self: flex-start; /* Align label to the left */
             margin-left: 5%; /* Adjust margin for alignment */
        }

        /* Classifier Radio Buttons */
        .stRadio [role="radiogroup"] {
            display: flex;
            flex-direction: row;
            justify-content: center;
            gap: 2rem; /* Spacing between radio buttons */
            margin-bottom: 2rem;
        }
        .stRadio label {
            color: white !important;
            background-color: rgba(255, 255, 255, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            transition: background-color 0.3s ease;
        }
        .stRadio input[type="radio"]:checked + div > label {
             background-color: rgba(255, 255, 255, 0.3);
        }


        /* Text Area Styling */
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 15px !important;
            border: none !important;
            min-height: 150px !important;
            padding: 1rem !important;
            font-size: 1rem !important;
            color: #333 !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            width: 90% !important; /* Make text area wider */
            margin: 0 auto; /* Center text area */
        }

        /* Button Styling */
        .stButton {
             width: 100%;
             display: flex;
             justify-content: center;
             margin-top: 1.5rem;
         }
        .stButton > button {
            border: none;
            border-radius: 25px;
            padding: 0.8rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            background: linear-gradient(90deg, #ff7e5f, #feb47b, #6dd5ed); /* Orange to Blue gradient */
            background-size: 200% auto;
            transition: background-position 0.5s ease;
            width: auto; /* Adjust width as needed */
            min-width: 200px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background-position: right center; /* change the direction of the change here */
            color: white;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .stButton > button:active {
            transform: translateY(1px);
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }

        /* Result styling */
        .stAlert {
             width: 90%;
             margin: 1rem auto;
             border-radius: 10px !important;
             text-align: center;
         }
    </style>
    """, unsafe_allow_html=True)

# --- Session State and Auth ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ''

DEMO_CREDENTIALS = { 'admin': 'admin123', 'user': 'user123' }

def login(username, password):
    if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

# --- Page Definitions ---
def show_login_page():
    login_css() # Apply login specific CSS
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<h1 class="login-title">Login</h1>', unsafe_allow_html=True)
    
    username = st.text_input("", placeholder="Username", key="username_input", label_visibility="collapsed")
    password = st.text_input("", placeholder="Password", type="password", key="password_input", label_visibility="collapsed")
    
    login_button_placeholder = st.empty()
    message_placeholder = st.empty()
    
    if login_button_placeholder.button("Login", key="login_button"):
        if login(username, password):
            login_button_placeholder.empty()
            st.rerun()
        else:
            # Show error message using markdown with class
            message_placeholder.markdown('<p class="error-message">Invalid username or password</p>', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True) # Close login-box and login-container


def show_main_page():
    main_dashboard_css()

    st.title("Select Classifier")
    classifier = st.radio(
        "Select Classifier", 
        ["Random Forest", "XGBoost"], 
        key="classifier_select", 
        label_visibility="hidden",
        horizontal=True
    )
    
    st.markdown("## Email Spam Detector", unsafe_allow_html=True)
    st.markdown("### Enter the content")
    
    try:
        # Load and preprocess data
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df.rename(columns={'v1': 'Category', 'v2': 'Message'})
        
        # Debug information
        st.sidebar.write("Dataset Info:")
        st.sidebar.write(f"Total samples: {len(df)}")
        st.sidebar.write(f"Spam samples: {len(df[df['Category'] == 'spam'])}")
        st.sidebar.write(f"Ham samples: {len(df[df['Category'] == 'ham'])}")
        
        # Enhanced text preprocessing
        import re
        def preprocess_text(text):
            # Convert to lowercase
            text = text.lower()
            # Replace URLs with 'url_link'
            text = re.sub(r'http\S+|www\S+|https\S+', 'url_link', text, flags=re.MULTILINE)
            # Replace email addresses with 'email_address'
            text = re.sub(r'\S+@\S+', 'email_address', text)
            # Replace currency symbols and numbers with 'money_amount'
            text = re.sub(r'[$€£¥]|\d+', 'money_amount', text)
            # Keep special characters that might be important for spam detection
            text = re.sub(r'[^\w\s!?.,$@]', ' ', text)
            # Add space around special characters
            text = re.sub(r'([!?.,@$])', r' \1 ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        # Apply preprocessing to dataset
        df['Message'] = df['Message'].apply(preprocess_text)
        
        # Enhanced feature extraction with TF-IDF and character n-grams
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.85,
            stop_words='english',
            ngram_range=(1, 3),  # Use up to trigrams
            analyzer='char_wb',  # Use character n-grams including word boundaries
            strip_accents='unicode',
            max_features=5000,  # Limit features to most important ones
            norm='l2'
        )
        
        X = vectorizer.fit_transform(df['Message'])
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Category'])
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize classifier with optimized parameters
        if classifier == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                class_weight={0: 1, 1: 3},  # Reverted to favor spam class slightly
                n_jobs=-1,
                criterion='gini'
            )
            
            # Add SMOTE for handling imbalanced data
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Train with resampled data
            model.fit(X_train_resampled, y_train_resampled)
        else:  # XGBoost
            import xgboost as xgb
            
            # Apply SMOTE for XGBoost as well
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train.toarray(), y_train)
            
            # Calculate sample weights favouring spam
            sample_weights = [3.0 if label == 1 else 1.0 for label in y_train_resampled]
            
            model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=3,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=1.5,  # Increased weight for spam class
                use_label_encoder=False,
                tree_method='hist',
                objective='binary:logistic'
            )
            
            # Train with resampled data and sample weights
            model.fit(
                X_train_resampled, 
                y_train_resampled,
                sample_weight=sample_weights
            )

        # User input section
        user_message = st.text_area("", height=150, key="user_message_input",
                                    placeholder="Enter your email content here...")
        
        # Placeholder for results
        result_placeholder = st.empty()

        if st.button("Analyze Email", key="analyze_email_button"):
            if user_message:
                # Preprocess user input
                processed_message = preprocess_text(user_message)
                user_input_vectorized = vectorizer.transform([processed_message])
                
                # Get prediction and probabilities
                if classifier == "XGBoost":
                    # Convert to dense array for XGBoost
                    user_input_dense = user_input_vectorized.toarray()
                    prediction = model.predict(user_input_dense)[0]
                    proba = model.predict_proba(user_input_dense)[0]
                else:
                    prediction = model.predict(user_input_vectorized)[0]
                    proba = model.predict_proba(user_input_vectorized)[0]
                
                # Adjust probability threshold based on classifier
                base_spam_threshold = 0.45 if classifier == "Random Forest" else 0.4  # Lowered thresholds
                
                # Show feature importance and spam indicators
                st.write("Analysis Details:")
                
                # Enhanced spam indicators with more patterns
                spam_indicators = []
                ham_indicators = []
                
                # Check message characteristics
                lower_msg = processed_message.lower()
                words = lower_msg.split()
                
                # Urgency indicators
                urgency_words = [
                    'urgent', 'immediate', 'important', 'action', 'required', 'verify', 'update', 'now',
                    'hurry', 'limited', 'expires', 'deadline', 'quick', 'fast', 'instant'
                ]
                urgency_count = sum(1 for word in urgency_words if word in lower_msg)
                if urgency_count >= 1:  # Relaxed condition (1 or more)
                    spam_indicators.append("Contains urgency keywords")
                
                # Banking/Financial indicators
                financial_words = [
                    'bank', 'account', 'credit', 'debit', 'card', 'verify', 'update', 'security',
                    'password', 'login', 'access', 'funds', 'transfer', 'transaction', 'balance'
                ]
                financial_count = sum(1 for word in financial_words if word in lower_msg)
                if financial_count >= 2:  # Relaxed condition (2 or more)
                    spam_indicators.append("Contains multiple banking/financial terms")
                
                # Link/Click indicators
                if 'url_link' in lower_msg and ('click' in lower_msg or 'link' in lower_msg):
                    spam_indicators.append("Contains suspicious link or click request") # Reverted to simpler condition
                
                # Security/Official claims
                official_words = [
                    'security', 'team', 'department', 'official', 'staff', 'support', 'service',
                    'administrator', 'system', 'notification', 'alert', 'warning', 'notice'
                ]
                official_count = sum(1 for word in official_words if word in lower_msg)
                if official_count >= 2:  # Relaxed condition (2 or more)
                    spam_indicators.append("Claims to be from official source")
                
                # Additional spam patterns
                if 'money_amount' in lower_msg and any(word in lower_msg for word in ['send', 'receive', 'transfer', 'payment']):
                    spam_indicators.append("Contains monetary amounts with transfer request")
                if 'email_address' in lower_msg and any(word in lower_msg for word in ['verify', 'confirm', 'click', 'update']):
                    spam_indicators.append("Contains email verification request")
                if len(re.findall(r'[!?]', lower_msg)) > 2: # Reverted to > 2
                    spam_indicators.append("Contains excessive punctuation")
                if len(re.findall(r'[A-Z\s]{10,}', user_message)) > 0: # Reverted to 10+
                    spam_indicators.append("Contains excessive capital letters")
                
                # Ham indicators
                if 5 < len(words) < 500:
                    ham_indicators.append("Normal message length")
                if len(set(words)) / len(words) > 0.6:
                    ham_indicators.append("Natural language diversity")
                if not any(char.isdigit() for char in user_message):
                    ham_indicators.append("No numerical content")
                if len(re.findall(r'[!?]', lower_msg)) <= 2:
                    ham_indicators.append("Normal punctuation usage")
                if not any(word in lower_msg for word in urgency_words):
                    ham_indicators.append("No urgency language")
                if sum(1 for word in financial_words if word in lower_msg) <= 1:
                    ham_indicators.append("Limited financial terms")
                if 'url_link' not in lower_msg and 'email_address' not in lower_msg:
                    ham_indicators.append("No URLs or email addresses")
                
                # Calculate adjusted spam score
                spam_score = len(spam_indicators) * 0.15   # Increased spam indicator weight
                ham_score = len(ham_indicators) * 0.1    # Decreased ham indicator weight
                
                # Adjust final prediction based on combined evidence
                if classifier == "Random Forest":
                    # Increase model probability weight again for RF
                    combined_spam_prob = (proba[1] * 0.85 + spam_score - ham_score)
                else:
                    combined_spam_prob = (proba[1] + spam_score - ham_score)
                
                combined_spam_prob = max(0, min(1, combined_spam_prob))  # Clamp between 0 and 1
                
                final_prediction = 'spam' if combined_spam_prob > base_spam_threshold else 'ham'
                
                # Show indicators
                if spam_indicators:
                    st.warning("⚠️ Suspicious patterns detected:")
                    for indicator in spam_indicators:
                        st.write(f"- {indicator}")
                
                if ham_indicators:
                    st.info("✅ Ham indicators detected:")
                    for indicator in ham_indicators:
                        st.write(f"- {indicator}")
                
                # Show prediction confidence
                st.write(f"\nSpam Score: {combined_spam_prob:.1%}")
                
                with result_placeholder.container():
                    st.write(f"Using {classifier} classifier:")
                    if final_prediction == 'spam':
                        st.error(f"🚨 This message is likely SPAM (Confidence: {combined_spam_prob:.1%})")
                        if combined_spam_prob > 0.8:
                            st.error("⚠️ High confidence spam detection!")
                    else:
                        st.success(f"✅ This message is likely HAM (Confidence: {(1-combined_spam_prob):.1%})")
            else:
                with result_placeholder.container():
                    st.warning("Please enter a message first.")

    except FileNotFoundError:
        st.error("Fatal Error: `spam.csv` not found. Please ensure the file exists in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        import traceback
        st.error(f"Stack trace: {traceback.format_exc()}")

# --- Main App Logic --- 
# Apply CSS based on authentication state
if st.session_state.authenticated:
    # If authenticated, show the main dashboard with its new CSS
    show_main_page()
else:
    # Otherwise, show the login page with its CSS (including bubbles)
    show_login_page() 