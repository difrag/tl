import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.pipeline import Pipeline

def plot_metrics(labels, preds, model_name):
    """Function to plot confusion matrix and return figure."""
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for ' + model_name)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['Negative', 'Positive'])
    ax.set_yticklabels([''] + ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig

def main():
    st.title('Dynamic ML App')
    st.write("Upload a CSV file. The app will automatically use the last column as the target for prediction, and add predictions as a new column.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the dataset
        st.write("Data Preview:")
        st.write(data.head(50))

        # Automatically determine if the target is categorical or continuous
        if data.iloc[:, -1].dtype == 'object':
            task_type = 'Classification'
        else:
            task_type = 'Regression'

        # Dynamic target and features setup
        X = data.iloc[:, :-1]  # all columns except the last one
        y = data.iloc[:, -1]   # the last column as the target

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing & model pipeline setup
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        # Define the models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'Classification' \
            else RandomForestRegressor(n_estimators=100, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=5)

        # Create and train the pipelines
        rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', rf_model)])
        knn_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', knn_model)])
        
        rf_clf.fit(X_train, y_train)
        knn_clf.fit(X_train, y_train)

        # Predictions and evaluation
        rf_pred = rf_clf.predict(X_test)
        knn_pred = knn_clf.predict(X_test)
        
        # Add predictions as new columns to the original data
        data['RF Predicted'] = rf_clf.predict(X)
        data['KNN Predicted'] = knn_clf.predict(X)

        # Tabs for results
        tab1, tab2 = st.tabs(["Random Forest", "K-Nearest Neighbors"])

        with tab1:
            st.write('Random Forest Results')
            if task_type == 'Classification':
                rf_score = accuracy_score(y_test, rf_pred)
                st.write('Accuracy:', rf_score)
                st.pyplot(plot_metrics(y_test, rf_pred, "Random Forest"))
            else:
                rf_score = mean_squared_error(y_test, rf_pred)
                st.write('MSE:', rf_score)
            st.write(data[['RF Predicted']].head(50))

        with tab2:
            st.write('K-Nearest Neighbors Results')
            if task_type == 'Classification':
                knn_score = accuracy_score(y_test, knn_pred)
                st.write('Accuracy:', knn_score)
                st.pyplot(plot_metrics(y_test, knn_pred, "K-Nearest Neighbors"))
            else:
                knn_score = mean_squared_error(y_test, knn_pred)
                st.write('MSE:', knn_score)
            st.write(data[['KNN Predicted']].head(50))
