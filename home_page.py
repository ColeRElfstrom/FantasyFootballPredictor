import streamlit as st
import pandas as pd
import numpy as np

from data_collection import return_full_df 
from data_collection import train
from data_collection import plot_player

df = return_full_df()

col_list = df["player_name"]

click = False

if 'count' not in st.session_state:
    st.session_state.count = 0


def run(player):
    format()
    st.session_state.count += 1
    metric_container = st.container()
    y_pred, y_test = train(player)
    metric_container.metric("Predicted Score: ", y_pred)
    metric_container.metric("Actual Score: ", y_test[0])
    plot = plot_player(player)
    st.pyplot(plot)


def format():
    st.title('Cole\'s Fantasy Football Predicitor')
    player = st.multiselect('Select', col_list)
    st.button('RUN', on_click=run, args=(player))


if st.session_state.count < 1:
    format()

