
import streamlit as st
import pandas as pd

# Create a sample DataFrame
data = pd.DataFrame({
    'Column 1': [1, 2, 3, 4],
    'Column 2': ['A', 'B', 'C', 'D']
})

# Display the DataFrame
st.dataframe(data)

# Display a table
st.table(data)
