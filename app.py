import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Water Contamination Detection System",
    page_icon="💧",
    layout="centered"
)

# ==============================
# LOAD MODEL
# ==============================

model = load_model("model.keras")

# ==============================
# TITLE
# ==============================

st.title("💧 Water Contamination Detection System")

st.markdown("""
This AI system predicts whether water is:

✅ Safe for Drinking  
❌ Contaminated  

using Deep Learning.
""")

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("Project Information")

st.sidebar.info("""
Deep Learning Based Real-Time Water Contamination Detection and Health Risk Prediction System
""")

# ==============================
# INPUT SECTION
# ==============================

st.header("Enter Water Quality Parameters")

ph = st.number_input("pH", value=7.0)

hardness = st.number_input("Hardness", value=200.0)

solids = st.number_input("Solids", value=10000.0)

chloramines = st.number_input("Chloramines", value=7.0)

sulfate = st.number_input("Sulfate", value=300.0)

conductivity = st.number_input("Conductivity", value=400.0)

organic_carbon = st.number_input("Organic Carbon", value=10.0)

trihalomethanes = st.number_input("Trihalomethanes", value=70.0)

turbidity = st.number_input("Turbidity", value=4.0)

# ==============================
# PREDICTION BUTTON
# ==============================

if st.button("Predict Water Quality"):

    # Prepare input
    input_data = np.array([[
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]])

    # Predict
    prediction = model.predict(input_data)

    prediction_value = prediction[0][0]

    contamination_percentage = (1 - prediction_value) * 100

    safe_percentage = prediction_value * 100

    # ==============================
    # RESULT SECTION
    # ==============================

    st.header("Prediction Result")

    if prediction_value > 0.5:

        st.success("✅ Water is Safe for Drinking")

        st.metric(
            label="Water Safety Percentage",
            value=f"{safe_percentage:.2f}%"
        )

        st.info("""
        Health Risk Level: LOW
        
        Water quality appears safe based on the provided parameters.
        """)

    else:

        st.error("❌ Water is Contaminated")

        st.metric(
            label="Contamination Percentage",
            value=f"{contamination_percentage:.2f}%"
        )

        st.warning("""
        Possible Health Risks:
        
        • Diarrhea  
        • Cholera  
        • Typhoid  
        • Skin Infection  
        """)

    # ==============================
    # BAR GRAPH
    # ==============================

    st.header("Water Parameter Analysis")

    parameters = [
        "pH",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic Carbon",
        "Trihalomethanes",
        "Turbidity"
    ]

    values = [
        ph,
        hardness,
        solids,
        chloramines,
        sulfate,
        conductivity,
        organic_carbon,
        trihalomethanes,
        turbidity
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(parameters, values)

    plt.xticks(rotation=45)

    plt.title("Water Quality Parameters")

    st.pyplot(fig)

# ==============================
# SHOW TRAINING GRAPHS
# ==============================

st.header("Model Performance Graphs")

try:
    st.image(
        "accuracy_graph.png",
        caption="Model Accuracy Graph"
    )

    st.image(
        "loss_graph.png",
        caption="Model Loss Graph"
    )

    st.image(
        "water_distribution.png",
        caption="Water Quality Distribution"
    )

except:
    st.warning("Training graphs not found.")

# ==============================
# FOOTER
# ==============================

st.markdown("---")

st.markdown("""
### Project Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
""")

st.markdown("""
Developed by Suraj Pandey

Under Guidance of HOD Dr. Tauseef Ahmad
""")