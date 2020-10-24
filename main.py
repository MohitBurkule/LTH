import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
@st.cache
def load_data(nrows):
	data = pd.read_csv('pune.csv', nrows=nrows)
	return data
components.html(
    """
	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script>
$('a[target="_blank"]').replaceWith("hi");
</script>
   """,
    height=600,
)
data=load_data(10).set_index('date_time')	
st.line_chart(data)