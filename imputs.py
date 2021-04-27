import pandas as pd
import datetime as dt
import statsmodels.api as sm
import streamlit as st
import numpy as np
from datetime import timedelta
import yfinance as yf




list_ = ['AAPL', 'XOM','GE','GS']


def get_list():
    st.sidebar.header('Welcome！ o(*￣▽￣*)ブ')
    symbol = st.sidebar.text_input("Input Tickers")
    st.sidebar.write("You can add **US** Tickers to the list.")
    st.sidebar.write("eg. Input **'MCD'** for US tickers.")
    if st.sidebar.button("Add Tickers"):
        list_.append(symbol.upper())
    drop = st.sidebar.selectbox("Drop a Ticker from the stock list.",np.sort(list_))
    if st.sidebar.button("Drop Tickers"):   
        list_.remove(drop)
    return list_

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    return next_month - dt.timedelta(days=next_month.day)

month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
def get_date():
    delta = timedelta(days=-1)
    col1,col2 = st.beta_columns([3,1])
    col3,col4 = st.beta_columns([3,1])
    with col1:
        start_year = st.selectbox('Selecting the Start Year', range(1990, 2022), index = 27)
    with col2:
        start_month = st.selectbox('Month', month_list, index = 5)
    with col3:
        end_year = st.selectbox('Selecting the End Year', range(1990, 2022), index = 30)
    with col4:
        end_month = st.selectbox('Month', month_list, index = 1)
    
    start_date = dt.date(start_year,month_list.index(start_month)+1,1) 
    end_date = last_day_of_month(dt.date(end_year, month_list.index(end_month)+1, 1))
    if start_date < end_date:
        st.success('Start Month: `%s` `%s`\n\nEnd Month:`%s` `%s`' % (start_year, start_month,end_year, end_month))
    else:
        st.error('Error: End Month must fall after start Month.')
    return start_date, end_date, start_year,start_month,end_year,end_month
