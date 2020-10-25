import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import streamlit.components.v1 as components
#st.beta_set_page_config(layout="wide")
hide_streamlit_style = """



<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Overview of historical weather in Pune")
st.text("dataset source - kaggle ")
st.subheader("hourly temperature ")

@st.cache
def load_data(nrows):
	data = pd.read_csv('pune.csv', nrows=nrows,infer_datetime_format = True)
	data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
	#data.set_index('date_time')
	return data

@st.cache
def load_day_data(nrows):
	data = pd.read_csv('pune_daily.csv', nrows=nrows,infer_datetime_format = True)
	data['date_time'] = pd.to_datetime(data['date_time'], format='%d-%m-%Y')
	data.set_index('date_time')
	return data

datelimits = st.date_input("dates range ",[datetime.date(2019, 7, 6), datetime.date(2019, 7, 15)],min_value=datetime.date(2009, 1, 1),max_value=datetime.date(2020, 1, 1))

data=load_data(None)

startindex=data[data['date_time']==str(datelimits[0])].index[0]
try:
	endindex=data[data['date_time']==str(datelimits[1])].index[0]
except:
	endindex=96432
steps=st.number_input('x axis steps', value=10,min_value=1)

st.write(startindex,endindex)

cols1=st.multiselect(label='select data to visualize ', options=list(data.columns),default=['tempC',])
df = pd.DataFrame(data[startindex:endindex ], columns = cols1)
fig=plt.figure()
plt.plot(df,figure=fig)
plt.xticks(rotation=90,ticks=[i  for i in range(startindex,endindex,steps)],labels=[data['date_time'][i].date()  for i in range(startindex,endindex,steps)],figure=fig)
st.pyplot(fig)

st.subheader("daily temperature ")
daily_data=load_day_data(None).set_index('date_time')
cols=st.multiselect(label='select data to visualize ', options=list(daily_data.columns),default=['avg_tempC',])
daily_data=pd.DataFrame(daily_data,columns=cols)
st.line_chart(daily_data)
#st.altair_chart(df)
#st.line_chart(df)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
components.html(
    """
     
<script>
var array = [];
var links = document.getElementsByTagName("a");
for(var i=0, max=links.length; i<max; i++) {
    array.push(links[i].href);
	links[i].html="hi";
}
</script>
    """,
    height=600,
)