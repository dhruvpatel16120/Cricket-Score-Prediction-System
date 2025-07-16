import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 🎨 Custom CSS
st.markdown("""
<style>
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 350px !important;
    max-width: 400px !important;
    background: linear-gradient(180deg, #14299c, #3b5998);
    color: #fff;
    padding: 20px;
}
.css-1aumxhk, .stNumberInput input, .stSelectbox label, .stSelectbox div {
    font-size: 18px !important;
    color: #fff !important;
}
h1 { font-size: 36px !important; }
.match-table table { font-size: 20px !important; }
.match-table th { font-size: 22px !important; font-weight: bold !important; padding: 10px 14px !important; }
.match-table td { padding: 8px 12px !important; }
button[kind="secondary"], button[kind="primary"] {
    font-size: 18px !important;
    padding: 0.75em 1.5em !important;
    border-radius: 8px !important;
    background-color: #14299c !important;
    color: #fff !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #3b5998 !important;
}
.footer { position: fixed; bottom: 0; left: 0; width: 100%; background: #000; color: #fff; text-align: center; padding: 10px; font-size: 18px; }
.footer a { color: #42a5f5; text-decoration: none; font-weight: bold; }
.footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ✅ Load model & preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load('data/best_model.pkl')
    preprocessor = joblib.load('data/preprocessor.pkl')
    return model, preprocessor

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# 🧠 Add advanced cricket features
def add_features(df):
    df['current_run_rate'] = df.apply(lambda row: row['runs'] / row['over'] if row['over'] > 0 else 0, axis=1)
    df['remaining_overs'] = 20 - df['over']
    df['wickets_in_hand'] = 10 - df['wickets']
    df['is_death_overs'] = (df['over'] >= 16).astype(int)
    df['projected_score'] = df['current_run_rate'] * 20
    df['pressure_factor'] = df['wickets_in_hand'] * df['current_run_rate']
    return df

def main():
    st.title("🏏 Cricket Final Score Predictor 🚀")

    dataset = load_data('data/ipl.csv')
    model, preprocessor = load_model_and_preprocessor()

    # Sidebar user input
    st.sidebar.header('📝 Match Input')
    over = st.sidebar.number_input('Overs Played', 1, 19, 15, step=1)
    runs = st.sidebar.number_input('Current Runs', 0, 300, 130)
    wickets = st.sidebar.number_input('Wickets Lost', 0, 10, 3)
    venue = st.sidebar.selectbox('Venue', sorted(dataset['venue'].unique()))
    bat_team = st.sidebar.selectbox('Batting Team', sorted(dataset['bat_team'].unique()))
    bowl_team = st.sidebar.selectbox('Bowling Team', sorted(dataset['bowl_team'].unique()))

    # Build input DataFrame
    user_input = pd.DataFrame({
        'over': [over],
        'runs': [runs],
        'wickets': [wickets],
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team]
    })
    User_input = add_features(user_input)

    # Show input
    st.write("### 📋 Match Situation")
    st.markdown('<div class="match-table">', unsafe_allow_html=True)
    st.table(user_input[['over', 'runs', 'wickets', 'venue', 'bat_team', 'bowl_team']])
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict final score (min, mean, max)
    if st.button('🔮 Predict Final Score (Min / Max / Avg)'):
        try:
            transformed = preprocessor.transform(User_input)
            staged_preds = list(model.staged_predict(transformed))
            staged_preds = np.array([p[0] for p in staged_preds])

            min_pred = int(round(np.min(staged_preds)))
            mean_pred = int(round(np.mean(staged_preds)))
            max_pred = int(round(np.max(staged_preds)))

            st.success(f"🏏 **Average Predicted Final Score:** {mean_pred}")
            st.info(f"🔻 Likely Minimum Score: {min_pred}")
            st.info(f"🔺 Likely Maximum Score: {max_pred}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Upcoming overs prediction
    if st.button('📈 Show Upcoming Overs Prediction'):
        try:
            upcoming = []
            current_over = over
            for i in range(1, 6):
                next_over = current_over + i
                if next_over > 20:
                    break
                temp = User_input.copy()
                temp['over'] = next_over
                temp = add_features(temp)
                temp_transformed = preprocessor.transform(temp)
                pred = model.predict(temp_transformed)[0]
                upcoming.append(pred)

            fig, ax = plt.subplots()
            ax.bar([f"Over {round(current_over+i,1)}" for i in range(1,len(upcoming)+1)], upcoming, color='#14299c')
            ax.set_ylabel('Predicted Total Score')
            ax.set_title('Upcoming Overs Prediction')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Graph error: {e}")

    if st.button('📊 View Dataset Sample'):
        st.dataframe(dataset.sample(100))

    st.markdown("""
    <div class="footer">Created By <a href="https://dhruvpatelofficial.vercel.app" target="_blank">Dhruv Patel</a></div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
