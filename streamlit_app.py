import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="KMeans and Decision Tree Application",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a title and description
st.title("Ανέβασε τα ντάτα σου εδώ για να τη βρεις")
st.write("Upload a tab-separated TXT file (no header) and explore the data using KMeans clustering and Decision Tree classification.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a file")

# Fucntion to make string values numerical
data2 = data.iloc[1:, :-1] # Dont use first row and last column
def encode_string_columns_to_num(data2):  # Dont use first row and last column
    label_encoders = {}
    for column in data.columns:
        if data2[column].dtype == 'object' or data2[column].dtype == 'string' :
            le = LabelEncoder()
            data2[column] = le.fit_transform(data2[column])
            label_encoders[column] = le
    return data2, label_encoders

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file, sep=',', header=None)
        else:
            data = pd.read_excel(uploaded_file, sep=',', header=None)
    except Exception as e:
        st.error(f"Error processing file: {e}")
    # Display a preview of the data
    st.subheader("Data Preview")
    st.write(data.head())
    #encode strings into nums for kmean to work properly
    data, _ = encode_string_columns_to_num(data)
    # Display a preview of the changed data
    st.subheader("Num Data Preview")
    st.write(data2.head())

# Create a sidebar for model options
st.sidebar.title("Model Configuration")

# Create number input widgets for KMeans and Decision Tree
n_clusters = st.sidebar.number_input("Number of Clusters for KMeans", min_value=2, value=2)
max_depth = st.sidebar.number_input("Max Depth for Decision Tree", min_value=1, value=3)


# Functions to run KMeans and Decision Tree
def run_kmeans(data, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    score = silhouette_score(data, labels)
    return labels, score

def run_decision_tree(X, y, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Run the analysis when the user clicks the button
if st.button("Start Analysis"):
    # Separate the features and target
    features = data.iloc[1:, :-1] # Dont use first row and last column
    target = data.iloc[:, -1]     # label = target (?)

    # Run KMeans and Decision Tree
    kmeans_labels, kmeans_score = run_kmeans(features, n_clusters)
    dt_accuracy = run_decision_tree(features, target, max_depth)

    # Display the evaluation results
    st.subheader("Evaluation Results")
    results = pd.DataFrame({
        "Method": ["KMeans (Silhouette Score)", "Decision Tree (Accuracy)"],
        "Score": [kmeans_score, dt_accuracy]
    })
    st.write(results)


