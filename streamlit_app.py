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

def plot_metrics(labels, preds, model_name,):
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots() 
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix fosr ' + model_name)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['Negative', 'Positive'])
    ax.set_yticklabels([''] + ['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig

def main():
    st.title('Dynamic ML App')
    st.write("Upload a .csv or .xls file. Attention: The app will automatically use the last column as the target for prediction")

    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Data Preview:")
        st.write(data.head(50))

        # Determine task type
        if data.iloc[:, -1].dtype == 'object':
            task_type = 'Classification'
        else:
            task_type = 'Regression'

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'Classification' \
            else RandomForestRegressor(n_estimators=100, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=5)

        rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', rf_model)])
        knn_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', knn_model)])
        
        rf_clf.fit(X_train, y_train)
        knn_clf.fit(X_train, y_train)

        rf_pred = rf_clf.predict(X_test)
        knn_pred = knn_clf.predict(X_test)
        
        data['RF Predicted'] = rf_clf.predict(X)
        data['KNN Predicted'] = knn_clf.predict(X)

        tab1, tab2 = st.tabs(["κατηγοριοποίηση - classificiation ", "ομαδοποίηση - clustering"])

        with tab1:
            st.write('Random Forest Results')

            if task_type == 'Classification':
                rf_score = accuracy_score(y_test, rf_pred)
                st.write('Accuracy:', rf_score)
            else:
                rf_score = mean_squared_error(y_test, rf_pred)
                st.write('MSE:', rf_score)
            st.write('K-Nearest Neighbors Results')

            if task_type is 'Classification':
                knn_score = accuracy_score(y_test, knn_pred)
                st.write('Accuracy:', knn_score)
            else:
                knn_score = mean_squared_error(y_test, knn_pred)
                st.write('MSE:', knn_score)

            # Using st.columns to create a side-by-side layout
            col1, col2 = st.columns(2)

            with col1:
                st.pyplot(plot_metrics(y_test, rf_pred, "Random Forest",))

            with col2:
                st.write("Random Forest Predictions vs Actual")
                st.write(data.iloc[:, [-3,-2]])

        with tab2:
            

            # Side-by-side layout using columns
            col1, col2 = st.columns(2)

            
                

if __name__ == "__main__":
    main()
