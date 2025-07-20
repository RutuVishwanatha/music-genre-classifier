import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder
model = joblib.load('models/genre_classifier.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF67;'>🎶 Music Genre Classifier 🎶</h1>
    <p style='text-align: center;'>Predict the genre of music based on audio features.</p>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Enter Audio Features")

with st.form("genre_form"):
    st.write("### Adjust the sliders for each feature")

    tempo = st.slider('🎵 Tempo', min_value=40, max_value=220, step=1, value=120)
    mfcc1_mean = st.slider('🎚️ MFCC1 Mean', min_value=-500.0, max_value=500.0, step=1.0, value=0.0)
    mfcc2_mean = st.slider('🎚️ MFCC2 Mean', min_value=-500.0, max_value=500.0, step=1.0, value=0.0)
    mfcc3_mean = st.slider('🎚️ MFCC3 Mean', min_value=-500.0, max_value=500.0, step=1.0, value=0.0)

    st.markdown("### Optional: Your Expected Genre")

    genre_options = ['blues', 'classical', 'country', 'disco', 'hiphop',
                     'jazz', 'metal', 'pop', 'reggae', 'rock',
                     'EDM', 'folk', 'indie', 'latin', 'kpop']

    user_genre = st.selectbox("🎧 Select Expected Genre:", genre_options)

    submit_button = st.form_submit_button("🎵 Predict Genre")


if submit_button:
    input_data = {
        'tempo': tempo,
        'mfcc1_mean': mfcc1_mean,
        'mfcc2_mean': mfcc2_mean,
        'mfcc3_mean': mfcc3_mean
    }

    input_df = pd.DataFrame(input_data, index=[0])

    if input_df.isnull().values.any():
        st.error("❌ Please fill in all the features to make a prediction.")
    else:
        prediction = model.predict(input_df)
        predicted_genre = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎉 **Predicted Genre:** {predicted_genre}")
        st.markdown("---")
        st.info(f"🎧 **Your Selected Expected Genre:** {user_genre}")
