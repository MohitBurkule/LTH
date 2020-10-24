import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache
def load_data(nrows):
	data = pd.read_csv('pune.csv', nrows=nrows)
	return data

data=load_data(10).set_index('date_time')	
st.line_chart(data)