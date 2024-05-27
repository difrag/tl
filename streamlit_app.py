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
from matplotlib.ticker import FixedLocator, FixedFormatter
import seaborn as sns
import os

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

    tick_labels = list(set(labels))
    tick_labels.sort()  # Ensure the labels are sorted for proper alignment to avoid UserWarning: FixedFormatter should only be used together with FixedLocator
    
    # Setting tick positions and labels
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.xaxis.set_major_locator(FixedLocator(range(len(tick_labels))))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.yaxis.set_major_locator(FixedLocator(range(len(tick_labels))))
    ax.yaxis.set_major_formatter(FixedFormatter(tick_labels))

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    return fig

# Function to plot clustering results
def plot_clusters(data, labels, title):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(data.shape[1])])
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

# Main function for Streamlit app
def main():
    st.title("Run your data through our Machine Learning App")
    st.markdown('''This app demonstrates certain data preprocessing and analysis techniques using Streamlit.
                Upload a CSV or Excel file to preprocess the data, apply label encoding to categorical features, and visualize the results.
                The app includes classification, clustering, and 2D visualization tabs for exploring the data.''')
    
    uploaded_file = st.file_uploader("Upload your file:", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        
        with st.sidebar:
            st.header("Parameter Controls")
            n_clusters = st.slider("Number of Clusters (Clustering)", 2, 10, 3)
            n_estimators = st.slider("Number of Estimators (Random Forest)", 10, 500, 100)
            eps = st.slider("DBSCAN Epsilon", 0.1, 1.0, 0.5)
            min_samples = st.slider("DBSCAN Min Samples", 1, 10, 5)

        try:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()
            

            if file_extension == 'csv':
                data = pd.read_csv(uploaded_file, header=None)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(uploaded_file, header=None)
            else:
                st.write("Error: Unsupported file type.")
                return

            st.write("Data Preview:")
            st.write(data.head(20))

        except Exception as e:

            st.error(f"Error reading file: {e}")
            return

        categorical_features = data.select_dtypes(include=['object']).columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns

        preprocessed_data = apply_label_encoding(data, categorical_features)

        

        if not preprocessed_data.empty:
            X = preprocessed_data.iloc[:, :-1]
            y = preprocessed_data.iloc[:, -1]

            unique_classes = y.nunique()
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            tab3, tab1, tab2, tab4, tab5, tab0 = st.tabs(["2D Visualization", "Classification", "Clustering", "Comparing of results", "Relevant information", "Debug Tab"])

            with tab0:
                st.write(f"Uploaded file name: {uploaded_file.name}")
                st.write(f"Uploaded file info: {uploaded_file}")
                st.write(f"File extension: {file_extension}")
                st.write("Categorical Features:", list(categorical_features))
                st.write("Numeric Features:", list(numeric_features))
                st.write("Number of Unique Classes:", unique_classes)
                st.write("Xtrain is ",X_train,X_test)
                st.write("ytrain  is ",y_train, y_test)
                st.write("Shape of X_train_scaled:", X_train_scaled.shape)
                st.write("Type of X_train_scaled:", type(X_train_scaled))
                st.write("scalled vs nonscaled",X_train_scaled,X_train)
                
            with tab1:
                rf_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                rf_clf.fit(X_train_scaled, y_train)
                rf_preds = rf_clf.predict(X_test_scaled)

                rf_accuracy = accuracy_score(y_test, rf_preds)
                rf_f1 = f1_score(y_test, rf_preds, average='weighted')
                rf_precision = precision_score(y_test, rf_preds, average='weighted')
                rf_recall = recall_score(y_test, rf_preds, average='weighted')

                st.pyplot(plot_confusion_matrix(y_test, rf_preds, "Random Forest"))

                st.write("Random Forest Metrics:")
                st.write("Accuracy:", rf_accuracy)
                st.write("F1-Score:", rf_f1)
                st.write("Precision:", rf_precision)
                st.write("Recall:", rf_recall)

                st.markdown("---------------------")
              
                lr_clf = LogisticRegression()
                lr_clf.fit(X_train_scaled, y_train)
                lr_preds = lr_clf.predict(X_test_scaled)
                lr_accuracy = accuracy_score(y_test, lr_preds)
                lr_f1 = f1_score(y_test, lr_preds, average='weighted')
                lr_precision = precision_score(y_test, lr_preds, average='weighted')
                lr_recall = recall_score(y_test, lr_preds, average='weighted')

                st.pyplot(plot_confusion_matrix(y_test, lr_preds, "Logistic Regression"))

                st.write("Logistic Regression Metrics:")
                st.write("Accuracy:", lr_accuracy)
                st.write("F1-Score:", lr_f1)
                st.write("Precision:", lr_precision)
                st.write("Recall:", lr_recall)


            with tab2:
                if isinstance(X_train_scaled, np.ndarray) and X_train_scaled.size > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans_labels = kmeans.fit_predict(X_train_scaled)
                        
                    kmeans_silhouette = silhouette_score(X_train_scaled, kmeans_labels)
                    kmeans_inertia = kmeans.inertia_

                    st.pyplot(plot_clusters(X_train_scaled, kmeans_labels, "KMeans Clustering"))

                    st.write("KMeans Clustering Metrics:")
                    st.write("Silhouette Score:", kmeans_silhouette)
                    st.write("Inertia:", kmeans_inertia)

                    st.markdown("-----")

                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan_labels = dbscan.fit_predict(X_train_scaled)
                    if len(set(dbscan_labels)) > 1:  # Ensure DBSCAN has formed clusters
                        dbscan_silhouette = silhouette_score(X_train_scaled, dbscan_labels)

                        st.pyplot(plot_clusters(X_train_scaled, dbscan_labels, "DBSCAN Clustering"))

                        st.write("DBSCAN Clustering Metrics:")
                        st.write("Silhouette Score:", dbscan_silhouette)
                        st.write("Inertia is not a good metric to use for evaluating DBSCAN models")
                    else:
                        st.write("DBSCAN did not form valid clusters.")

                    # else:
                    #      st.write("Error: Cannot perform clustering. X_train_scaled is empty or invalid.")

            with tab3:
                st.title("Exploratory Data Analysis (EDA)")

                st.write("Statistical Summary:")
                st.write(generate_statistical_summary(preprocessed_data))
                st.markdown("---------------------")

                st.write("Missing Values:")
                st.write(check_missing_values(preprocessed_data))
                st.markdown("---------------------")

                st.header("Data Distribution:")
                st.write("Distribution plots help assess the skewness, kurtosis, and presence of outliers in the data. They are essential for understanding the data's characteristics, identifying potential data transformations, and selecting appropriate statistical models.")
                for feature in numeric_features:
                    fig, ax = plt.subplots()
                    sns.histplot(data[feature], kde=True, ax=ax)
                    ax.set_title(f'Distribution of feature {feature}')
                    st.pyplot(fig)

                st.markdown("---------------------")
                
                correlation_matrix = preprocessed_data.corr()
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
                st.write("Correlation heatmaps are useful for feature selection, identifying multicollinearity (high correlation between features), and understanding the underlying structure of the data")
                st.markdown("---------------------")

                st.header("Dimensionality Reduction Techniques")
                st.write("Dimensionality reduction techniques aim to reduce the number of features (dimensions) in a dataset while preserving its essential structure and information. They  help visualize and explore high-dimensional data, uncover hidden patterns, reduce noise, and improve model generalization. ")

                pca = PCA(n_components=2)
                pca_transformed = pca.fit_transform(X_train_scaled)

                st.pyplot(plot_2d_visualization(pca_transformed, y_train, "PCA Visualization"))
                st.write("Principal Component Analysis (PCA): Projects high-dimensional data onto directions of maximum variance, useful for understanding overall data structure.")
                

                tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
                tsne_transformed = tsne.fit_transform(X_train_scaled)
                
                st.markdown("---------------------")    

                st.pyplot(plot_2d_visualization(tsne_transformed, y_train, "t-SNE Visualization"))
                st.write("t-Distributed Stochastic Neighbor Embedding (t-SNE): Projects high-dimensional data into 2D, useful for visualizing patterns and clusters.")

                st.markdown("------------")
                # if unique_classes > 1:
                #     max_components = min(X_train.shape[1], unique_classes - 1) ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
                #     lda_components = min(max_components, 2) ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
                #     lda = LinearDiscriminantAnalysis(n_components=2)
                #     lda_transformed = lda.fit_transform(X_train_scaled, y_train)

                #     st.pyplot(plot_2d_visualization(lda_transformed, y_train, "LDA Visualization"))
                #     st.write("Linear Discriminant Analysis (LDA): A supervised linear technique that maximizes class separation, ideal for visualizing class distinctions.")
                # else:
                #     st.write("LDA not available. Requires at least two classes.")
                            #### Kept getting n_components cannot be larger than min(n_features, n_classes - 1) error in LDA ####
                # try:
                #     lda = LinearDiscriminantAnalysis(n_components=2)  # Set n_components to 2
                #     lda_transformed = lda.fit_transform(X_train_scaled, y_train)

                #     st.pyplot(plot_2d_visualization(lda_transformed, y_train, "LDA Visualization"))
                #     st.write("Linear Discriminant Analysis (LDA): A supervised linear technique that maximizes class separation, ideal for visualizing class distinctions.")
                # except ValueError as e:
                #     st.error(f"Error during LDA transformation: {e}")
                #     st.write("Number of unique classes:", unique_classes)
                #     st.write("Number of features (n_features):", X_train_scaled.shape[1])

                # st.markdown("---------------------")

                isomap = Isomap(n_components=2, n_neighbors=10)
                isomap_transformed = isomap.fit_transform(X_train_scaled)

                st.pyplot(plot_2d_visualization(isomap_transformed, y_train, "Isomap Visualization"))
                st.write("Isomap: A nonlinear technique that seeks to preserve geodesic distances, useful for discovering complex structures and manifolds.")
            
            with tab4:
                st.title("Comparing Results")
                st.header("Performance Metrics Comparison")
                metrics_dict = {
                    "Random Forest": {"Accuracy": rf_accuracy, "F1-Score": rf_f1},
                    "Logistic Regression": {"Accuracy": lr_accuracy, "F1-Score": lr_f1},
                    "KMeans Clustering": {"Silhouette Score": kmeans_silhouette},
                    "DBSCAN Clustering": {"Silhouette Score": dbscan_silhouette}
                }

                metrics_df = pd.DataFrame(metrics_dict).T
                st.write(metrics_df)
                st.header("Best Algorithm")

                st.markdown("------------")
                # Calculate composite score for each algorithm
                composite_scores = {}
                for algorithm in metrics_dict.keys():
                    composite_score = metrics_df.loc[algorithm].mean()
                    composite_scores[algorithm] = composite_score
                    st.write(f"Composite Score for {algorithm}: {composite_score}")
                    
                # Determine the best algorithm based on the highest composite score
                best_algorithm = max(composite_scores, key=composite_scores.get)

                st.markdown("------------")
                st.write(f"The best algorithm based on composite scores is {best_algorithm}")

                # Plot performance metrics
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df.plot(kind="bar", ax=ax)
                plt.title("Performance Metrics Comparison")
                plt.xlabel("Algorithm")
                plt.ylabel("Score")
                plt.xticks(rotation=35)
                st.pyplot(fig)

            with tab5:
                st.markdown("""
                    ### Info

                    **Features:**
                    - **Upload and Preview:** Upload a CSV or Excel file and preview the data.
                    - **Data Preprocessing:** Automatically handle categorical features by applying label encoding.
                    - **Model Training:** Train and evaluate classification models (Random Forest and Logistic Regression) and visualize the results.
                    - **Clustering:** Apply KMeans and DBSCAN clustering algorithms and visualize the clusters.
                    - **Dimensionality Reduction:** Perform PCA, t-SNE and Isomap for 2D visualization of high-dimensional data.

                    **How it Works:**
                    - **File Upload:** Upload your data file. The application reads and preprocesses the data.
                    - **Model Training:** Choose the parameters for Random Forest and Logistic Regression.
                    - **Clustering:** Select the number of clusters for KMeans or adjust parameters for DBSCAN(epsilon and minimum samples).
                    - **Visualization:** View 2D representations of your data using PCA, t-SNE  or Isomap.

                    **Development Team:**
                    - **Dimitris Fragkoulis:** Π2015191
                    - **Konstantinos Nikolopoulos :** Π2016***
                    - **Christos Grigorakos:** Π201****

                    **Contact Information:**
                    - **Email:** p15frag@ionio.gr
                    - **Github:** [Dimitris](https://github.com/difrag), [Konstantinos](), [Christos]()
                    """)      
        else:
            st.markdown("**Please upload a file to continue.**")

if __name__ == "__main__":
    main()
