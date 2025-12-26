import streamlit as st
import joblib

# load the trained model (tuned Linear SVM pipeline)
model = joblib.load("best_fake_review_svm.joblib")

st.set_page_config(page_title="Review Detector", page_icon="📝")

st.title("Review Detector")
st.write(
    "Paste a product review below and click **Predict** to see whether the model "
    "thinks it is *AI-generated* or *human-written*."
)

review_text = st.text_area("Review text:", height=200)

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        pred = model.predict([review_text])[0]

        if pred == "ai":
            st.error("Prediction: **AI-generated (fake) review**")
        else:
            st.success("Prediction: **Human-written (original) review**")

        st.caption(
            "Note: the model is trained on one specific fake-review dataset, "
            "so predictions may not generalise to all types of text."
        )
