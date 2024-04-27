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
st.title("Taaaaabular Data Analysis with KMeans and Decision Tree")
st.write("Upload a CSV or Excel file with tabular data, and the app will transform text data into numerical values, show the transformed data, and apply KMeans clustering and Decision Tree classification.")


# Function to make string values numerical
def read_and_one_hot_encode(uploaded_file):
    if uploaded_file is not None:
        try:
            # Attempt to read the uploaded file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
                st.write(data)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
                st.write(data)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
            
            # Check if the DataFrame is empty
            if data.empty:
                st.error("The uploaded file is empty. Please upload a file with data.")
                return None
            
            # Check for any missing values in the DataFrame
            if data.isnull().values.any():
                st.warning("The uploaded file contains missing values. "
                           "These may need to be addressed before applying machine learning techniques.")
                # One option is to fill missing values, or you can drop them
                # data = data.fillna(method='ffill')  # Example: forward fill
                # data = data.dropna()  # Example: drop rows with missing values
            
            # Select columns that are not of numeric type
            non_numeric_columns = data.select_dtypes(exclude=['number']).columns
            
            # Apply one-hot encoding to non-numeric columns
            data = pd.get_dummies(data, columns=non_numeric_columns)
            
            # Return the one-hot encoded DataFrame
            return data
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            return None
    else:
        # If no file was uploaded, inform the user
        st.info("Please upload a file to proceed.")
        return None

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
processed_data = read_and_one_hot_encode(uploaded_file)

if processed_data is not None:
    st.write(processed_data)
    # Create a sidebar for model options
    st.sidebar.title("Model Configuration")
    # Create number input widgets for KMeans and Decision Tree
    n_clusters = st.sidebar.number_input("Number of Clusters for KMeans", min_value=2, value=2)
    max_depth = st.sidebar.number_input("Max Depth for Decision Tree", min_value=1, value=3)
    # Separate the features and target
    target_column = st.selectbox("Select the target column", processed_data.columns)
    features = processed_data.drop(target_column, axis=1)
    target = processed_data[target_column]
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
