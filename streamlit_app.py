import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt
import os.path


# Function to label encode categorical features
def apply_label_encoding(data, categorical_columns):
    le = LabelEncoder()
    encoded_data = data.copy()  # Make a copy of the original data
    for col in categorical_columns:
        if encoded_data[col].dtype == 'object':
            encoded_data[col] = le.fit_transform(encoded_data[col])
    return encoded_data

# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + list(set(labels)))
    ax.set_yticklabels([''] + list(set(labels)))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    return fig

# Function to plot clustering results
def plot_clusters(data, labels, title):
    # Ensure data is a DataFrame
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(data.shape[1])])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.colorbar(scatter)
    return fig

# Main function for Streamlit app
def main():
    st.title("Streamlit Data Preprocessing App")

    # Sidebar controls for key parameters
    with st.sidebar:
        st.header("Parameter Controls")
        n_clusters = st.slider("Number of Clusters (Clustering)", 2, 10, 3)  # Default is 3 clusters
        n_estimators = st.slider("Number of Estimators (Random Forest)", 10, 500, 100)  # Default is 100 estimators
        eps = st.slider("DBSCAN Epsilon", 0.1, 1.0, 0.5)  # DBSCAN epsilon parameter
        min_samples = st.slider("DBSCAN Min Samples", 1, 10, 5)  # DBSCAN minimum samples parameter

    # Step 1: File upload and handling
    st.write("Upload a CSV or Excel file to preprocess.")
    uploaded_file = st.file_uploader("Upload your file:", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Step 2: Read the uploaded file based on extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".csv":
            data = pd.read_csv(uploaded_file)  # Read CSV file
        elif file_extension in [".xls", ".xlsx"]:
            data = pd.read_excel(uploaded_file, engine='openpyxl')  # Read Excel file with openpyxl engine

        # Display a preview of the data
        st.write("Data Preview:")
        st.write(data.head())

        # Step 3: Identify categorical and numerical features
        categorical_features = data.select_dtypes(include=['object']).columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        
        st.write("Categorical Features:", list(categorical_features))
        st.write("Numeric Features:", list(numeric_features))

        # Step 4: Apply label encoding to categorical features
        preprocessed_data = apply_label_encoding(data, categorical_features)

        # Step 5: Display the preprocessed data
        st.write("Preprocessed Data:")
        st.write(preprocessed_data.head())

        # Step 6: Further processing and analysis
        if not preprocessed_data.empty:
            # Separate features and target (assuming the last column is the target)
            X = preprocessed_data.iloc[:, :-1]  # Features
            y = preprocessed_data.iloc[:, -1]  # Target

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply scaling to numeric features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Step 7: Basic visualization
            tab1, tab2 = st.tabs(["Classification", "Clustering"])

            # Classification tab
            with tab1:
                # Random Forest Classifier with user-defined estimators
                rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                rf_clf.fit(X_train_scaled, y_train)

                rf_preds = rf_clf.predict(X_test_scaled)

                # Metrics for Random Forest
                rf_accuracy = accuracy_score(y_test, rf_preds)
                rf_f1 = f1_score(y_test, rf_preds, average='weighted')
                rf_precision = precision_score(y_test, rf_preds, average='weighted')
                rf_recall = recall_score(y_test, rf_preds, average='weighted')

                st.write("Random Forest Metrics:")
                st.write("Accuracy:", rf_accuracy)
                st.write("F1-Score:", rf_f1)
                st.write("Precision:", rf_precision)
                st.write("Recall:", rf_recall)

                st.pyplot(plot_confusion_matrix(y_test, rf_preds, "Random Forest"))

                # Logistic Regression Classifier
                lr_clf = LogisticRegression()
                lr_clf.fit(X_train_scaled, y_train)

                lr_preds = lr_clf.predict(X_test_scaled)

                # Metrics for Logistic Regression
                lr_accuracy = accuracy_score(y_test, lr_preds)
                lr_f1 = f1_score(y_test, lr_preds, average='weighted')
                lr_precision = precision_score(y_test, lr_preds, average='weighted')
                lr_recall = recall_score(y_test, lr_preds, average='weighted')

                st.write("Logistic Regression Metrics:")
                st.write("Accuracy:", lr_accuracy)
                st.write("F1-Score:", lr_f1)
                st.write("Precision:", lr_precision)
                st.write("Recall:", lr_recall)

                st.pyplot(plot_confusion_matrix(y_test, lr_preds, "Logistic Regression"))

            # Clustering tab
            with tab2:
                # Clustering with KMeans using user-defined clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans_labels = kmeans.fit_predict(X_train_scaled)

                # Metrics for KMeans Clustering
                kmeans_silhouette = silhouette_score(X_train_scaled, kmeans_labels)
                kmeans_inertia = kmeans.inertia_

                st.write("KMeans Clustering Metrics:")
                st.write("Silhouette Score:", kmeans_silhouette)
                st.write("Inertia:", kmeans_inertia)

                st.pyplot(plot_clusters(X_train_scaled, kmeans_labels, "KMeans Clustering"))

                # Clustering with DBSCAN with user-defined parameters
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan_labels = dbscan.fit_predict(X_train_scaled)

                # Metrics for DBSCAN Clustering
                dbscan_silhouette = silhouette_score(X_train_scaled, dbscan_labels)

                st.write("DBSCAN Clustering Metrics:")
                st.write("Silhouette Score:", dbscan_silhouette)

                st.pyplot(plot_clusters(X_train_scaled, dbscan_labels, "DBSCAN Clustering"))
            
    else:
        st.write("Please upload a file to continue.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
