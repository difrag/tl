import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

def main():
    st.title('Dynamic ML App')
    st.write("Upload a CSV file. The app will automatically use the last column as the target for prediction and add predictions as a new column.")

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

        # Define the model
        if task_type == 'Classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Create and train the pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = clf.predict(X_test)
        # Add predictions as a new column to the original data
        data['Predicted'] = clf.predict(X)  # Predict on the entire dataset for a new column

        if task_type == 'Classification':
            score = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', score)
        else:
            score = mean_squared_error(y_test, y_pred)
            st.write('MSE:', score)

        # Show updated DataFrame
        st.write("Updated Data with Predictions:")
        st.write(data.head(50))

if __name__ == "__main__":
    main()
