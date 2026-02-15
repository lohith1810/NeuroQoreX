import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.ensemble import RandomForestClassifier

timepoints = np.arange(1, 13)
msrcs_synthetic = np.array([0.85, 0.82, 0.78, 0.72, 0.65, 0.58, 
                            0.55, 0.60, 0.68, 0.75, 0.80, 0.84])

def closed_loop_controller(value):
    if value >= 0.75:
        return "Normal", 0
    elif 0.60 <= value < 0.75:
        return "Mild Stress", 5
    elif 0.50 <= value < 0.60:
        return "Moderate Stress", 10
    else:
        return "Severe Stress", 15

states, doses = zip(*[closed_loop_controller(v) for v in msrcs_synthetic])

traj_df = pd.DataFrame({
    "Timepoint": timepoints,
    "MSRCS": msrcs_synthetic,
    "State": states,
    "Recommended Dose (mg)": doses
})

tau_level = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
msrcs_from_tau = np.clip(1 - tau_level, 0, 1)

tau_df = pd.DataFrame({
    "Tau_Level": tau_level,
    "MSRCS_Score": msrcs_from_tau,
})

np.random.seed(17)
n_patients = 47
msrcs_data = pd.DataFrame({
    'Mito_ROS': np.random.normal(0.5, 0.15, n_patients),
    'Redox_Imbalance': np.random.normal(0.5, 0.15, n_patients),
    'ATP_Decline': np.random.normal(0.4, 0.1, n_patients),
    'Oxidative_Stress': np.random.normal(0.6, 0.2, n_patients)
})

msrcs_data['Risk'] = ((msrcs_data['Mito_ROS'] > 0.6) | 
                      (msrcs_data['Redox_Imbalance'] > 0.6)).astype(int)

features = ['Mito_ROS', 'Redox_Imbalance', 'ATP_Decline', 'Oxidative_Stress']
feat_matrix = msrcs_data[features]
targets = msrcs_data['Risk']

model = RandomForestClassifier(n_estimators=100, random_state=17)
model.fit(feat_matrix, targets)

new_patients = pd.DataFrame({
    'Patient_ID': ['P-2847', 'P-1923', 'P-5601'],
    'Mito_ROS': [0.65, 0.4, 0.55],
    'Redox_Imbalance': [0.7, 0.35, 0.5],
    'ATP_Decline': [0.45, 0.3, 0.55],
    'Oxidative_Stress': [0.6, 0.4, 0.65]
})

predictions = model.predict(new_patients[features])
new_patients['Predicted_Risk'] = predictions
new_patients['Therapy_Suggestion'] = new_patients['Predicted_Risk'].apply(
    lambda r: "Initiate BDNF-GABA Protocol" if r == 1 else "Monitor / No Intervention"
)

max_neurons = 15
neuron_matrix = np.zeros((len(timepoints), max_neurons))

for t_idx, score in enumerate(msrcs_synthetic):
    neuron_count = 5 + int(score * 10)
    for n in range(neuron_count):
        neuron_matrix[t_idx, n] = score 

neuron_df = pd.DataFrame(neuron_matrix, columns=[f"N{i+1}" for i in range(max_neurons)])
neuron_df['Timepoint'] = timepoints
activity_long = neuron_df.melt(id_vars='Timepoint', var_name='Neuron', value_name='Activity')

app = Dash(__name__)

graph_template = "plotly_dark"
card_style = {'backgroundColor': '#2E2E2E', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}

app.layout = html.Div(style={'backgroundColor': '#121212', 'color': 'white', 'fontFamily': 'system-ui, Segoe UI, sans-serif', 'padding': '40px'}, children=[
    
    html.Div([
        html.H1("NeuroQoreX MSRCS Dashboard", style={'textAlign': 'center', 'color': '#6B8CAE', 'marginBottom': '10px'}),
        html.P("Biomarker Analysis & Closed-Loop Control", style={'textAlign': 'center', 'color': '#AAAAAA'}),
    ], style={'marginBottom': '40px'}),

    html.Div(style=card_style, children=[
        html.H3("Longitudinal MSRCS Trajectory", style={'color': '#6B8CAE'}),
        dcc.Graph(
            figure=px.line(traj_df, x="Timepoint", y="MSRCS", markers=True, template=graph_template, title="Patient Stability Over Time")
            .add_hline(y=0.75, line_dash="dash", line_color="#4A7C59", annotation_text="Normal")
            .add_hline(y=0.60, line_dash="dash", line_color="#C4A35A", annotation_text="Mild Stress")
            .add_hline(y=0.50, line_dash="dash", line_color="#B85450", annotation_text="Critical")
        )
    ]),

    html.Div([
        html.Div(style={**card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}, children=[
            html.H3("Tau vs MSRCS Correlation", style={'color': '#6B8CAE'}),
            dcc.Graph(
                figure=px.scatter(tau_df, x="Tau_Level", y="MSRCS_Score", trendline="ols", template=graph_template,
                                  title="Tau vs Cognitive Score")
            )
        ]),
        html.Div(style={**card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}, children=[
            html.H3("Risk Prediction", style={'color': '#6B8CAE'}),
            dcc.Graph(
                figure=px.bar(new_patients, x="Patient_ID", y="Predicted_Risk",
                              color="Predicted_Risk", color_continuous_scale=['#4A7C59', '#B85450'],
                              template=graph_template, title="Risk Assessment")
            )
        ])
    ]),

    html.Div(style=card_style, children=[
        html.H3("Patient Triage List", style={'color': '#6B8CAE'}),
        dcc.Graph(
            figure=go.Figure(data=[go.Table(
                header=dict(values=list(new_patients.columns),
                            fill_color='#333333', font=dict(color='white'), align='left'),
                cells=dict(values=[new_patients[k] for k in new_patients.columns],
                           fill_color='#222222', font=dict(color='white'), align='left'))
            ]).update_layout(template=graph_template, margin=dict(l=0, r=0, t=0, b=0), height=150)
        )
    ]),

    html.Div(style=card_style, children=[
        html.H3("Neural Activity Heatmap", style={'color': '#6B8CAE'}),
        dcc.Graph(
            figure=px.density_heatmap(activity_long, x='Timepoint', y='Neuron', z='Activity',
                                      color_continuous_scale='Plasma', template=graph_template,
                                      title="Simulated Neuron Firing Rates")
        )
    ])
])

if __name__ == "__main__":
    print("NeuroQoreX system online.")
    print("Access the dashboard here: http://127.0.0.1:8050/")
    app.run(debug=True)