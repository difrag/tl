# Import necessary libraries
import streamlit as st
import pandas as pd

# Create a Streamlit title
st.title("Data Preview Application")

# Create a file uploader widget for uploading a tab-separated TXT file
uploaded_file = st.file_uploader("Upload a tab-separated TXT file", type="txt")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file as a DataFrame using pandas
    data = pd.read_csv(uploaded_file, sep='\t', header=None)

    # Display the first 10 rows of the data
    st.write("Data preview (first 10 rows):")
    st.write(data.head(10))


# Add a title to the sidebar
st.sidebar.title("Sidebar Title")

# Add a text input widget to the sidebar
sidebar_input = st.sidebar.text_input("Enter some text in the sidebar:")

# Display the sidebar input in the main area
st.write(f"Your sidebar input: {sidebar_input}")

