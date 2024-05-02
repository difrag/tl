import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, f1_score
from sklearn.pipeline import Pipeline
#import seaborn as sns
import os.path 

# Function to plot confusion matrix
def plot_confusion_matrix(labels, preds, model_name):
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    fig.colorbar(cax)
    ax.set_xticklabels(['', 'Negative', 'Positive'])
    ax.set_yticklabels(['', 'Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig

# Function to plot clustering results
def plot_clusters(data, labels, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.colorbar(scatter, ax=ax)
    return fig

# Main function
def main():
    st.title("Dynamic ML App")
    st.write("Upload a CSV or Excel file.")
    st.write("Attention: The app will automatically use the last column as the target for prediction.")
    st.write("Attention: The app has no NaN or missing value handling capabilities (for now).")
    uploaded_file = st.file_uploader("Upload a file and run analysis ", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".csv":
            data = pd.read_csv(uploaded_file)  # Read CSV file
        elif file_extension in [".xls", ".xlsx"]:
            data = pd.read_excel(uploaded_file, engine='openpyxl')  # Read Excel files with openpyxl

        # Sidebar for user input
        with st.sidebar:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)  
            n_estimators = st.slider("Number of Estimators (Random Forest)", 10, 500, 100) 
            n_neighbors = st.slider("Number of Neighbors (KNN)", 1, 20, 5)  

        # Display data preview
        st.write("Data Preview:")
        st.write(data.head(50))

        # Determine task type based on the last column
        if data.iloc[:, -1].dtype == 'object':
            task_type = 'Classification'
        else:
            task_type = 'Regression'
        
        # Separate features and target
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Preprocessing pipelines
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Apply the preprocessing pipeline to features
        preprocessed_X = preprocessor.fit_transform(X)

        # Ensure dense data if needed
        if hasattr(preprocessed_X, "toarray"):
            preprocessed_X = preprocessed_X.toarray()

        # Convert to DataFrame with correct preprocessed data
        preprocessed_df = pd.DataFrame(preprocessed_X)  # Corrected assignment

        # Ensure it's not empty
        if preprocessed_df.empty:
            st.write("Error: Preprocessed DataFrame is empty.")

        # Correct assignment of features and target
        X = preprocessed_df.iloc[:, :-1]  # Features
        y = data.iloc[:, -1]  # Original target (not processed)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define classification models
        rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)  # Using slider value
        knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)  # Using slider value

        # Define clustering models
        spectral_clust = SpectralClustering(n_clusters=n_clusters, random_state=42)  # Using slider value
        hierarchical_clust = AgglomerativeClustering(n_clusters=n_clusters)

        # Fit classification models
        rf_clf.fit(X_train, y_train)  # Classification
        knn_clf.fit(X_train, y_train)  # Classification

        # Fit clustering models
        spectral_labels = spectral_clust.fit_predict(X)  #clustering
        hierarchical_labels = hierarchical_clust.fit_predict(X)  #clustering

        # Predictions for classification
        rf_pred = rf_clf.predict(X_test)  # Predictions from Random Forest
        knn_pred = knn_clf.predict(X_test)  # Predictions from KNN

        # Create tabs for classification and clustering
        tab1, tab2, tab3 = st.tabs(["2D Visualization", "Classification", "Clustering"])

        with tab1:
            st.write("Dimensionality Reduction and Visualization")
        with tab2:
            rf_accuracy = accuracy_score(y_test, rf_pred)  # Accuracy of Random Forest
            rf_f1 = f1_score(y_test, rf_clf.predict(X_test), average='weighted')  # F1 score of Random Forest
            st.write("Random Forest Accuracy:", rf_accuracy)
            st.write("Random Forest F1-Score:", rf_f1)
            st.pyplot(plot_confusion_matrix(y_test, rf_pred, "Random Forest"))

            
            knn_accuracy = accuracy_score(y_test, knn_pred)  # Accuracy of KNN
            knn_f1 = f1_score(y_test, knn_clf.predict(X_test), average='weighted')
            st.write("K-Nearest Neighbors Accuracy:", knn_accuracy)
            st.write("K-Nearest Neighbors F1-Score:", knn_f1)
            st.pyplot(plot_confusion_matrix(y_test, knn_pred, "K-Nearest Neighbors"))

        # Results for clustering
        with tab3:
            spectral_silhouette = silhouette_score(X, spectral_labels)  # Spectral silhouette score of Spectral clustering
            st.write("Spectral Clustering Silhouette Score:", spectral_silhouette)
            st.pyplot(plot_clusters(X, spectral_labels, "Spectral Clustering"))  

            hierarchical_silhouette = silhouette_score(X, hierarchical_labels)  # Hierarchical silhouette score
            st.write("Hierarchical Clustering Silhouette Score:", hierarchical_silhouette)
            st.pyplot(plot_clusters(X, hierarchical_labels, "Hierarchical Clustering"))  
            

if __name__ is "__main__":
    main()
