
import torch
import dash_mantine_components as dmc

from dash import Dash, html, callback, Input, State, Output
from transformers import AutoTokenizer, RobertaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./models/finetuned_roberta_model")
model = RobertaForSequenceClassification.from_pretrained(
    "./models/finetuned_roberta_model", torch_dtype=torch.float16, device_map="cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "HAWKISH", 1: "DOVISH", 2: "NEUTRAL"}

app = Dash(
    __name__,
    title="FEDDIE"
)

# Requires Dash 2.17.0 or later
app.layout = dmc.MantineProvider(
    [html.Div([
        dmc.TextInput(
            label="Enter your query:",
            id="query-field",
            size="sm",
            radius="sm"
        ),
        dmc.Button(
            "Submit",
            id="submit-query-button",
            variant="filled",
            size="sm",
            radius="sm",
        )]
    ),
        html.Div(
        id='output-field'
    )]
)


@callback(
    Output('output-field', 'children'),
    Input('submit-query-button', 'n_clicks'),
    State('query-field', 'value'),
    prevent_initial_call=True
)
def submit_query(n_clicks, query):
    inputs = tokenizer(query, return_tensors="pt",
                       padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    sentiment = id2label.get(predicted_class_id, f"Class {predicted_class_id}")
    return f"Predicted sentiment: {sentiment}"


if __name__ == '__main__':
    app.run(debug=True)
