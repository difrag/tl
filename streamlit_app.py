import streamlit as st

# Display a title
st.title("Hello, Streamlit!")

# Display a subtitle
st.subheader("This is a subtitle")

# Display plain text
st.write("This is some plain text.")



Παράδειγμα 2:

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
