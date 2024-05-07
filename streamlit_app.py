import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import matplotlib.pyplot as plt
import os.path
import seaborn as sns

# Function to generate statistical summaries
def generate_statistical_summary(data):
    return data.describe()

# Function to check for missing values
def check_missing_values(data):
    return data.isnull().sum()

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
        st.header("DATADATADATA",data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.colorbar(scatter)
    return fig

# Function to plot dimensionality reduction results
def plot_2d_visualization(data, labels, title):
    if data.shape[1] < 2:
        st.write("Error: Not enough dimensions for 2D visualization.")
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    fig.colorbar(scatter)
    return fig


####################################
#### Main function for Streamlit app
###
##
#
def main():
    st.title("Streamlit Data Preprocessing and machine learning App")
    st.header("Data Preprocessing and Machine Learning Analysis")
    #Markdown description of app
    st.markdown("This app demonstrates data preprocessing and analysis techniques using Streamlit. Upload a CSV or Excel file to preprocess the data, apply label encoding to categorical features, and visualize the results. The app includes classification, clustering, and 2D visualization tabs for exploring the data.")
    #st.set_option('deprecation.showPyplotGlobalUse', False) #To future-proof Streamlit apps, they recommend creating a figure explicitly with plt.subplots() and passing it to st.pyplot().
    
    # Sidebar controls for key parameters
    with st.sidebar:
        st.header("Parameter Controls")
        n_clusters = st.slider("Number of Clusters (Clustering)", 2, 10, 3)  # Default is 3 clusters
        n_estimators = st.slider("Number of Estimators (Random Forest)", 10, 500, 100)  # Default is 100 estimators
        eps = st.slider("DBSCAN Epsilon", 0.1, 1.0, 0.5)  # DBSCAN epsilon parameter
        min_samples = st.slider("DBSCAN Min Samples", 1, 10, 5)  # DBSCAN minimum samples parameter

    # File upload and handling
    uploaded_file = st.file_uploader("Upload your file:", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if uploaded_file.name:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension == '.csv':
                data = pd.read_csv(uploaded_file)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(uploaded_file)
                # Display a preview of the data########################################
                st.write("Data Preview:")
                st.write(data.head())
        else:
            st.write("Error: The uploaded file has no name or an invalid name. OR you need to Drag it again")
    
        
        # Identify categorical and numerical features
        categorical_features = data.select_dtypes(include=['object']).columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        
        st.write("Categorical Features:", list(categorical_features))
        st.write("Numeric Features:", list(numeric_features))

        # Apply label encoding to categorical features
        preprocessed_data = apply_label_encoding(data, categorical_features)

        # Display the preprocessed data
        st.write("Preprocessed Data:")
        st.write(preprocessed_data.head())

        # Further processing and analysis
        if not preprocessed_data.empty:
            # Separate features and target (assuming the last column is the target)
            X = preprocessed_data.iloc[:, :-1]  # Features
            y = preprocessed_data.iloc[:, -1]  # Target

            # Check the number of unique classes for LDA
            unique_classes = y.nunique()
            st.write("Number of Unique Classes:", unique_classes)

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply scaling to numeric features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                X_train_scaled
            except NameError:
                st.write("X_train_scaled is not defined")

            # Step 2: Check the shape and type of X_train_scaled
            st.write("Shape of X_train_scaled:", X_train_scaled.shape)
            st.write("Type of X_train_scaled:", type(X_train_scaled))


            ########
            ###Tabs
            ##
            #
            tab1, tab2, tab3 = st.tabs(["Classification", "Clustering", "2D Visualization"])

            # Classification tab
            with tab1:
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
                st.write("IS X_train_scaled INSTANCE  OF NDARRAY ?", isinstance(X_train_scaled, np.ndarray)) ##############################################
                st.write("IS X_train_scaled EMPTY ?", X_train_scaled.size > 0) ##############################################$##############################################
                if isinstance(X_train_scaled, np.ndarray) and X_train_scaled.size > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    st.write(kmeans) ##############################################
                    st.write("X_train_scaled",X_train_scaled) ##############################################
                    st.write("Type of X_train_scaled:", type(X_train_scaled)) ##############################################
                    kmeans_labels = kmeans.fit_predict(X_train_scaled)
                    
                    st.write("kmeans_labels:", kmeans_labels) ##############################################
                    kmeans_silhouette = silhouette_score(X_train_scaled, kmeans_labels)
                    kmeans_inertia = kmeans.inertia_

                    st.write("KMeans Clustering Metrics:")
                    st.write("Silhouette Score:", kmeans_silhouette)
                    st.write("Inertia:", kmeans_inertia)

                    st.pyplot(plot_clusters(X_train_scaled, kmeans_labels, "KMeans Clustering"))

                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan_labels = dbscan.fit_predict(X_train_scaled)

                    dbscan_silhouette = silhouette_score(X_train_scaled, dbscan_labels)

                    st.write("DBSCAN Clustering Metrics:")
                    st.write("Silhouette Score:", dbscan_silhouette)

                    st.pyplot(plot_clusters(X_train_scaled, dbscan_labels, "DBSCAN Clustering"))
                else:
                 
                   st.write("EError: Cannot perform clustering. X_train_scaled is empty or invalid.")

            # 2D Visualization tab
            with tab3:
                st.write("Exploratory Data Analysis (EDA)")

                # Statistical summaries
                st.write("Statistical Summary:")
                st.write(generate_statistical_summary(preprocessed_data))

                # Missing values
                st.write("Missing Values:")
                st.write(check_missing_values(preprocessed_data))

                # Data distribution plots
                st.write("Data Distribution:")
                for feature in numeric_features:
                    fig, ax = plt.subplots()
                    sns.histplot(data[feature], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {feature}')
                    st.pyplot(fig)

                # Correlation heatmap
                st.write("Correlation Heatmap:")
                correlation_matrix = preprocessed_data.corr()
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)


                st.write("Dimensionality Reduction Techniques")

                # PCA Visualization
                pca = PCA(n_components=2)  # Principal Component Analysis
                pca_transformed = pca.fit_transform(X_train_scaled)

                st.write("Principal Component Analysis (PCA): Projects high-dimensional data onto directions of maximum variance, useful for understanding overall data structure.")
                st.pyplot(plot_2d_visualization(pca_transformed, y_train, "PCA Visualization"))

                # t-SNE Visualization
                tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)  # t-SNE with default parameters
                tsne_transformed = tsne.fit_transform(X_train_scaled)

                st.write("t-Distributed Stochastic Neighbor Embedding (t-SNE): Projects high-dimensional data into 2D, useful for visualizing patterns and clusters.")
                st.pyplot(plot_2d_visualization(tsne_transformed, y_train, "t-SNE Visualization"))

                # Linear Discriminant Analysis (LDA) Visualization

                
                if unique_classes >2:  # Ensure there are enough classes for LDA
                    lda_components = min(X_train.shape[1], unique_classes - 1)  # Limit components
                    lda = LinearDiscriminantAnalysis(n_components=lda_components)
                    lda_transformed = lda.fit_transform(X_train_scaled, y_train)

                    st.write("Linear Discriminant Analysis (LDA): A supervised linear technique that maximizes class separation, ideal for visualizing class distinctions.")
                    st.pyplot(plot_2d_visualization(lda_transformed, y_train, "LDA Visualization"))
                else:
                    st.write("LDA not available. Requires at least two classes.")

                # Isomap Visualization
                isomap = Isomap(n_components=2, n_neighbors=10)  # Isomap with 10 neighbors
                isomap_transformed = isomap.fit_transform(X_train_scaled)

                st.write("Isomap: A nonlinear technique that seeks to preserve geodesic distances, useful for discovering complex structures and manifolds.")
                st.pyplot(plot_2d_visualization(isomap_transformed, y_train, "Isomap Visualization"))

    else:
        st.markdown("**Please upload a file to continue.**")

# Run the Streamlit app
if __name__ == "__main__":
    main()
