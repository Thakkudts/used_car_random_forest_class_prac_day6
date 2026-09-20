import gradio as gr
import joblib
import pandas as pd
import os

# Load trained Random Forest model
model = joblib.load("used_car_random_forest.pkl")


def predict_price_category(car_age, kilometers_driven):

    input_data = pd.DataFrame({
        "Car_Age": [car_age],
        "Kilometers_Driven": [kilometers_driven]
    })

    prediction = model.predict(input_data)[0]

    return f"Predicted Price Category: {prediction}"


demo = gr.Interface(
    fn=predict_price_category,

    inputs=[
        gr.Number(
            label="Enter Car Age (years)",
            minimum=0,
            maximum=30,
            value=3
        ),

        gr.Number(
            label="Enter Kilometers Driven",
            minimum=0,
            maximum=500000,
            value=30000
        )
    ],

    outputs=gr.Textbox(
        label="Predicted Price Category"
    ),

    title="🚗 Used Car Price Prediction",

    description="Predict the price category of a used car using Car Age and Kilometers Driven."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
