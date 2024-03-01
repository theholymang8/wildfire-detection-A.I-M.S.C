# Wildfire Detection Project

## Overview

This project leverages machine learning to develop an advanced fire detection system. By analyzing sensor data from various environmental conditions, including both fire and non-fire events, we aim to enhance the accuracy and reliability of fire detection in real-world settings.

## Objective

Our goal is to use machine learning to accurately identify fire events from environmental sensor data. The ability to distinguish between actual fires and normal variations in environmental conditions is crucial for creating an effective and reliable fire detection system.

## Dataset

The dataset includes sensor readings related to environmental conditions, collected to simulate the variety of scenarios that a real-world fire detection system might encounter. This approach ensures our model is trained on data that reflects both fire and everyday environmental conditions, enhancing its ability to function accurately in diverse settings.

## Methodology

1. **Data Preprocessing**: Cleaning and normalizing the data to ensure consistency and reliability in the analysis.
2. **Feature Selection**: Identifying the most impactful features for fire detection to improve model performance.
3. **Model Training**: Evaluating different machine learning models to find the most effective approach for fire detection.
4. **Evaluation and Optimization**: Assessing model performance with a focus on accuracy, precision, recall, and the F3 score, aiming to balance detection capabilities with minimizing false alarms.

## Results

The project explores various models, emphasizing the importance of accurate and reliable fire detection. Our methodology highlights the potential for machine learning to improve fire detection systems, making them more responsive and trustworthy in real-life applications.

## Conclusion

This analysis demonstrates the viability of using machine learning for fire detection, emphasizing the enhancement of detection systems for real-world applicability. Future work will focus on refining these models and exploring additional features to further improve their performance.

## Repository Structure

- `/data`: Contains the data sets for model training and evaluation.
- `/models`: Stores the trained machine learning models.
- `/notebooks`: Jupyter notebooks for detailed analysis, including model training and evaluation reports.
- `demo.py`: A Streamlit application for visualizing the model's predictions.

## Installation

### Clone the Repository

To get started with this project, clone the repository to your local machine:

```bash
git clone https://github.com/your-repo/wildfire-detection.git
cd wildfire-detection 
```

### Create a Virtual Environment

Set up a virtual environment to manage the projects dependencies.

For Unix/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

## Install Dependencies

Activate the virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```


# Usage
## Jupyter Notebooks

The analysis and reports can be found in the Jupyter notebooks within the /notebooks folder.
## Running the Streamlit App

To view the Streamlit dashboard locally, run the following command:

```bash
streamlit run demo.py
```

# Additional Information

The project's progress, including detailed reports and evaluation metrics, is documented in the Jupyter notebooks. For a comprehensive analysis, refer to the notebooks located in the /notebooks folder. The necessary data for model evaluation and testing is available in the /data directory.

We are committed to improving our model and welcome any feedback or contributions to the project.