# Streamlit Demo 
import pandas as pd
import streamlit as st
import plotly.express as px

#Page Config
st.set_page_config(page_title='My Dashboard', page_icon=':bar_chart:', layout='wide')
st.title('Modelacion de Sistemas 2025-2')

#read data
df = pd.read_csv('2025.csv', encoding='latin', delimiter=';')

# sidebar filters
selected_category = st.sidebar.selectbox('Select Category', df['NOMBRE INSTITUCIÓN'].unique())
selected_modo = st.sidebar.selectbox('Select Modo', df['JORNADA'].unique())
filtered_df1 = df[df['NOMBRE INSTITUCIÓN'] == selected_category]
filtered_df = filtered_df1[filtered_df1['JORNADA'] == selected_modo]

# Create a bar chart
fig_bar = px.bar(df, x='NOMBRE INSTITUCIÓN', y='TOTAL MATRÍCULA', title='Category Values')
st.plotly_chart(fig_bar)

# Create a filtered bar chart
fig_barf = px.bar(filtered_df, x='NOMBRE CARRERA', y='TOTAL MATRÍCULA', title=f"Category {selected_category}")
st.plotly_chart(fig_barf)

#show data
st.write(f"Data for {selected_category}:")
st.dataframe(filtered_df)