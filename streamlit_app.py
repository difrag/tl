import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file, sep=',', header=None)
        else:
            data = pd.read_excel(uploaded_file, sep=',', header=None)
    except Exception as e:
        st.error(f"Error processing file: {e}")
st.write("Data preview:")
st.dataframe(df.head())
# Display a preview of the data
st.subheader("Data Preview(1)")
st.write(data.head())


    # Create a sidebar for model options
    st.sidebar.title("Model Options")

    # Create number input widgets for KMeans and Decision Tree
    n_clusters = st.sidebar.number_input("Number of Clusters for KMeans", min_value=2, value=2)
    max_depth = st.sidebar.number_input("Max Depth for Decision Tree", min_value=1, value=3)

    # Run the analysis when the user clicks the button
    if st.button("Start Analysis"):
        # Separate the features and target
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]

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


