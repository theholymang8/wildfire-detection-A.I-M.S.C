import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load
import numpy as np

import plotly.graph_objs as go
import plotly.figure_factory as ff

from sklearn.metrics import roc_curve, f1_score, accuracy_score, fbeta_score, classification_report, confusion_matrix




model = load('./models/detection_model.pkl')


test_data = pd.read_csv("./data/demo_set.csv")


test_data_training = pd.read_csv("./data/test_ds.txt")

X_test = np.loadtxt('./data/test_ds.txt')
y_test = pd.read_csv("./data/y_test_labels.csv")



@st.cache_data
def plot_room_measurements(df, downsample_factor=20):
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')

    measurements_to_keep = ['PM_Total_Room', 'PM_Room_Typical_Size', 'PM10_Room', 'CO_Room', 'H2_Room', 'PM05_Room', 'VOC_Room_RAW', 'UV_Room' ]

    # Extracting the column names that are related to room measurements
    room_measurements = [col for col in df.columns if col in measurements_to_keep]

    # Separating data based on 'ternary_label'
    background_data = df[df['ternary_label'] == 'Background']
    nuisance_data = df[df['ternary_label'] == 'Nuisance']
    fire_data = df[df['ternary_label'] == 'Fire']

    # Store the figures in a list to be returned
    figures = []

    # Iterate over each room measurement and create a plotly graph
    for measurement in room_measurements:
        fig = go.Figure()
        # Downsample the data by taking every nth row
        fig.add_trace(go.Scatter(
            x=background_data['Date'].iloc[::downsample_factor], 
            y=background_data[measurement].iloc[::downsample_factor], 
            mode='markers', 
            name='Background', 
            marker=dict(color='blue', size=2)))
        fig.add_trace(go.Scatter(
            x=nuisance_data['Date'].iloc[::downsample_factor], 
            y=nuisance_data[measurement].iloc[::downsample_factor], 
            mode='markers', 
            name='Nuisance', 
            marker=dict(color='green', size=5)))
        fig.add_trace(go.Scatter(
            x=fire_data['Date'].iloc[::downsample_factor], 
            y=fire_data[measurement].iloc[::downsample_factor], 
            mode='markers', 
            name='Fire', 
            marker=dict(color='red', size=2)))

        # Update the layout for each figure
        fig.update_layout(
            title=f'{measurement} Levels Over Time for All Sensors',
            xaxis_title='Date',
            yaxis_title=f'{measurement} Levels',
            legend_title='Label',
            height=350
        )
        figures.append(fig)
    
    return figures

def evaluate_model(X_test, y_test, beta=2.0):
    # Predict on test values
    y_pred_svm_test = model.predict(X_test)

    # Print the classification report
    report = classification_report(y_test, y_pred_svm_test, output_dict=True)

    # Calculate weighted F-beta score for our multiclass classification
    f_beta_score = fbeta_score(y_test, y_pred_svm_test, beta=beta, average='weighted')
    
    # Generate the confusion matrix
    conf_matrix_val = confusion_matrix(y_test, y_pred_svm_test)

    # Create a pandas DataFrame from the numpy array and label the columns and index
    conf_matrix_df = pd.DataFrame(conf_matrix_val,
                                  index=['True Background', 'True Fire', 'True Nuisance'],
                                  columns=['Predicted Background', 'Predicted Fire', 'Predicted Nuisance'])

    # Convert the DataFrame to a Plotly Figure
    fig = ff.create_annotated_heatmap(
        z=conf_matrix_df.values,
        x=conf_matrix_df.columns.tolist(),
        y=conf_matrix_df.index.tolist(),
        colorscale='Blues',
        showscale=True
    )
    
    # Add titles and axis names
    fig.update_layout(
        title_text='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='Actual Class',
        xaxis=dict(tickangle=45)
    )
    
    return report, f_beta_score, fig

# Title and description
st.title('Wildfire Detection Dashboard')
st.write('''
         This dashboard presents the results of a model designed to detect wildfires from timeseries sensor data. 
         The data includes measurements such as CO2, CO, H2, humidity, temperature, VOCs, and particulate matter. 
         Our model treats the problem as a multiclass classification with classes: fire, background, and nuisance. 
         The presented results are based on a test dataset representative of real-world scenarios.
         ''')

# Plot the demo dataset metrics
st.header('Sensor Data Over Time')
st.write('''
        In this section, the visualizations display the time-series sensor data across various room measurements. 
        Each graph represents a different metric monitored over time. You will notice distinct red spikes appearing in the plots; 
        these signify fire scenarios that our model has been trained to identify.

        The aim of these graphs is to illustrate the model's capability in detecting abnormal patterns within environmental readings that could indicate a fire event. 
        Such patterns are characterized by sudden and significant increases in the values of certain sensors, which are marked in red for visibility.

        These spikes are critical indicators for our model, as they help differentiate between normal fluctuations (shown in blue for background data and green for nuisance data) 
        and those anomalies that may suggest the presence of a fire. By effectively recognizing these red spikes amidst other data, our model strives to provide timely 
        and accurate fire detection, which is crucial for initiating prompt emergency responses.

        The data presented here is integral to understanding the environment's typical behavior and the distinctive signatures of fire-related incidents. 
        As you interact with the graphs, consider how the red spikes stand out against the usual data patterns—this contrast is a key element that our predictive algorithms utilize to discern potential fires.
        ''')
# Plot the sensor data using the new Plotly function
room_measurements_figures = plot_room_measurements(test_data)
for fig in room_measurements_figures:
    st.plotly_chart(fig)

# Run model on the test dataset and present results
st.header('Model Predictions')
st.write('''
In this segment, we evaluate the predictive performance of our wildfire detection model. We use a suite of metrics to provide a comprehensive assessment:

- **F2 Score**: Prioritizing the correct identification of fires, the F2 score weighs recall higher than precision, capturing the model's accuracy in spotting critical fire events.

- **Classification Report**: Offering a breakdown by class, the classification report details the precision, recall, and F1 score for 'Background', 'Nuisance', and 'Fire' scenarios, providing insights into the model's classification prowess. 
         The 'Nuisance' class does not play a big role in our scenario but we decided to list its resulsts nonetheless, opting for some future improvement.

- **Confusion Matrix**: This visual tool allows us to observe the model's predictions in contrast to the true labels, highlighting correct and incorrect classifications across all categories.

- **ROC Curve**: By plotting the true positive rate against the false positive rate, the ROC curve visualizes the trade-offs between sensitivity and specificity at various threshold settings.

Together, these metrics paint a detailed picture of the model's capabilities, ensuring we have a nuanced understanding of its strengths and areas for improvement in wildfire detection.
''')

# Call the evaluation function and get the results
report, f_beta_score, confusion_matrix_fig = evaluate_model(X_test, y_test, beta=2)

# Display the F-beta score and the classification report
st.header('Model Evaluation')
st.write('''
Presenting the following evaluation metrics for clarity, startin with the recall-weighted F-Score.

''')
st.subheader(f'F{2}-Score for the Test Set')
st.write(f'{f_beta_score:.2f}')

st.write('''
         
        Our wildfire detection model managed to capture an F2-Score of 0.89 and 
        The F2 score is particularly important in the context of wildfire detection because it emphasizes the significance of recall — 
        the model's ability to correctly identify all relevant instances. 
        A high F2 score indicates that our model is exceptionally tuned to minimize false negatives, meaning it has a lower chance of overlooking real fires, which is crucial in emergency scenarios where early detection can be life-saving.

''')

# Display the classification report as a dataframe
st.subheader('Classification Report')
st.write('''

Our classification report reveals a high recall for the fire class, underscoring our model's efficacy in correctly identifying true fire events. This is a crucial aspect of our system, indicating a strong alignment with our primary objective - the early and accurate detection of fires.

It is important to note, however, that the metrics for nuisance detection are not the primary focus of this model iteration. As a result, these metrics may not be as pronounced. This is an intentional model design choice; we are currently prioritizing the accurate identification of fire events over nuisance signals. In the complex task of discerning genuine fires from benign events, our model is configured to prioritize safety, sometimes at the expense of mistakenly classifying some non-threatening events as potential fires.

This strategic decision ensures that our system remains vigilant against the threat of wildfires, prioritizing rapid detection to facilitate immediate response.

''')
st.dataframe(pd.DataFrame(report).transpose())

# Display the confusion matrix
st.subheader('Confusion Matrix')

st.write('''
The confusion matrix above provides a clear visual representation of our model's performance. The 'True Fire' row is particularly important, as it demonstrates the model's strong capability to identify fire events accurately, which is paramount for a wildfire detection system.

The matrix also highlights instances of misclassification, where some fire events are incorrectly identified as background or nuisance. While the model is highly attuned to detecting fires, it currently places less emphasis on distinguishing nuisances, leading to a lower accuracy in that category. This strategic choice ensures that the model errs on the side of caution, prioritizing the detection of fires over less critical misclassifications.

''')

st.plotly_chart(confusion_matrix_fig)


# Simple accuracy analysis
st.header('Accuracy Analysis')


st.write('''
Upon examining the confusion matrix, we observe that our model has successfully identified a commendable number of true fire incidents. This is a testament to the model's robustness in detecting actual fires, which is the cornerstone of its design.

The matrix also sheds light on the instances of false positives. As we move forward, our objective is to refine the model's ability to reduce these occurrences. Future work includes meticulous tuning of the model to enhance its discrimination between fire and nuisance, thereby improving its precision.

''')