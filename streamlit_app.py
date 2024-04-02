# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to run k-means clustering and calculate the silhouette score
def run_kmeans(data, k):
    kmeans = KMeans(n_clusters=k)  # Create a KMeans instance with the specified number of clusters
    kmeans.fit(data)               # Fit the KMeans model to the data
    labels = kmeans.labels_        # Get the cluster labels for each data point
    score = silhouette_score(data, labels)  # Calculate the silhouette score for the clustering
    return labels, score

# Function to run decision tree classification and calculate the accuracy
def run_decision_tree(X, y, max_depth):
    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    dt = DecisionTreeClassifier(max_depth=max_depth)  # Create a DecisionTreeClassifier instance with the specified max_depth
    dt.fit(X_train, y_train)       # Fit the classifier to the training data
    y_pred = dt.predict(X_test)    # Predict labels for the test data
    accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the classifier
    return accuracy

# Create a Streamlit title
st.title("KMeans and Decision Tree Application")

# Create a file uploader widget for uploading a tab-separated TXT file
uploaded_file = st.file_uploader("Upload a tab-separated TXT file (no header)", type="txt")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file as a DataFrame using pandas, assuming no header
    data = pd.read_csv(uploaded_file, sep='\t', header=None)
    
    # Display a preview of the data
    st.write("Data preview:")
    st.write(data.head())

   # DELETED FROM HERE<<<<

    # Create a button to start the analysis
    if st.button("Start Analysis"):
        # Separate the features (K-1 columns) and the target (last column) from the data
        features = data.iloc[:, :-1]
        target = data.iloc[:, -1]

        # Run k-means clustering and decision tree classification on the data
        kmeans_labels, kmeans_score = run_kmeans(features, n_clusters)
        dt_accuracy = run_decision_tree(features, target, max_depth)

        # Display the evaluation results in a table
        st.write("Evaluation Results:")
        st.write(pd.DataFrame({"Method": ["KMeans (Silhouette Score)", "Decision Tree (Accuracy)"],
                               "Score": [kmeans_score, dt_accuracy]}))


# Add a title to the sidebar
st.sidebar.title("Model options")


# Create a number input widget for specifying the number of clusters for k-means clustering
n_clusters = st.sidebar.number_input("Enter the number of clusters for KMeans:", min_value=2, value=2)

# Create a number input widget for specifying the max depth for the decision tree
max_depth = st.sidebar.number_input("Enter the max depth for the Decision Tree:", min_value=1, value=3)

# Display the sidebar input in the main area
st.write(f"Your sidebar input: {sidebar_input}")
