import pandas as pd

def make_prediction(model, input_data):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    return prediction, probability