import pandas as pd
import datetime as dt
import statsmodels.api as sm
import streamlit as st
import numpy as np
from datetime import timedelta
import yfinance as yf
from imputs import  get_date,get_list
import plotly.graph_objects as go




ASSETS = get_list()# get variables
ASSETS = list(dict.fromkeys(ASSETS))
ASSETS.sort()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

st.title("Fama French 3 Factors Analysis")
'''#### (stock prices from *yahoo finance*) '''
st.write("**Current Stock List**", list(dict.fromkeys(ASSETS)))

START_DATE,END_DATE, start_year,start_month,end_year,end_month = get_date()# get variables


df = yf.download(ASSETS, start=START_DATE, end=END_DATE)["Adj Close"]
df_rtn = df.pct_change()

def get_rtn():
    cal_retun = st.sidebar.selectbox("Select method to calculate returns",["Simple method", "Continuous method"])
    if cal_retun == "Simple method":
        #simple culculation for stock return
        rtn = df_rtn.dropna().resample('M').agg(lambda x: (1+x).prod()-1)
    elif cal_retun == "Continuous method": 
        #continous calculation for stock return
        rtn = np.log(df_rtn+1)
        rtn = rtn.dropna().resample('M').agg(lambda x: (1+x).prod()-1)
    return rtn


class Thereefactor_regression():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.alphas = []
        self.ts = []
        self.r2_adjs = []
        self.beta_1s = []
        self.beta_2s = []
        self.beta_3s = []
        self.r2s = []
        self.conf_ints = []
        self.mses = []
        self.resids = []
        X = sm.add_constant(X)
        for ind, col in enumerate(y.columns):
            a = y.iloc[:,ind]
            model = sm.OLS(a, X)
            results = model.fit()
            alpha, beta_1, beta_2, beta_3 = results.params
            t = results.tvalues
            r2_adj = results.rsquared_adj
            r2 = results.rsquared
            mse = results.mse_resid
            conf_int = results.conf_int()
            resid = results.resid
            self.alphas.append(alpha)
            self.beta_1s.append(beta_1)
            self.beta_2s.append(beta_2)
            self.beta_3s.append(beta_3)
            self.ts.append(t)
            self.r2_adjs.append(r2_adj)
            self.r2s.append(r2)
            self.resids.append(resid)
            self.conf_ints.append(conf_int)
            self.mses.append(mse)
        self.alpha_CI = [(self.conf_ints[i].round(decimals=5).iloc[0,:][0],self.conf_ints[i].round(decimals=5).iloc[0,:][1]) for i in range(len(y.columns))]
        self.beta_1_CI = [(self.conf_ints[i].round(decimals=5).iloc[1,:][0],self.conf_ints[i].round(decimals=5).iloc[1,:][1]) for i in range(len(y.columns))]
        self.beta_2_CI = [(self.conf_ints[i].round(decimals=5).iloc[2,:][0],self.conf_ints[i].round(decimals=5).iloc[2,:][1]) for i in range(len(y.columns))]
        self.beta_3_CI = [(self.conf_ints[i].round(decimals=5).iloc[3,:][0],self.conf_ints[i].round(decimals=5).iloc[3,:][1]) for i in range(len(y.columns))]
        #self.alpha_stats = [self.ts[i][0] for i in range(len(y.columns))]
        #self.beta_stats = [self.ts[i][1] for i in range(len(y.columns))]
        self.resids = pd.DataFrame(self.resids,index = y.columns).T #creat a residual dataframe
        self.sd_resid = (self.resids - self.resids.mean())/self.resids.std() #calculating the standize residual
    def summary_table(self):
        table = pd.DataFrame({"α":self.alphas
                            ,"95% CI for α":self.alpha_CI
                            ,"β1":self.beta_1s
                            ,"95% CI for β1":self.beta_1_CI
                            ,"β2":self.beta_2s
                            ,"95% CI for β2":self.beta_2_CI
                            ,"β3":self.beta_3s
                            ,"95% CI for β3":self.beta_3_CI
                            ,"R2": self.r2s
                            ,"R2_adj":self.r2_adjs
                            ,"SER":np.sqrt(self.mses)
                            },index = self.y.columns)
        return table




rtn = get_rtn()

#transform time
start = "{}".format(start_year) + "{0:02}".format(month_list.index(start_month)+1)
end = "{}".format(end_year) + "{0:02}".format(month_list.index(end_month)+1)

#get fama factors from downloaded database
factors = pd.read_csv('F-F_Research_Data_Factors.CSV', skiprows = 3, index_col = 0, nrows = 1137)
factors = factors.loc[start:end,:]
factors = factors/100

rtn.index = factors.index

#merge data
merge = pd.merge(rtn,factors,left_index=True, right_index=True)

#for loop calculating excess return
for i in range(len(ASSETS)):
    var = ASSETS[i]+"-"+"RF"
    merge[var] = merge.iloc[:,i] - merge["RF"]

y = merge.iloc[:,-len(ASSETS):]
X = factors.iloc[:,:3]

#get fama factors from pandas reader
#factors = reader.FamaFrenchReader('F-F_Research_Data_Factors.CSV')
#factors = factors.read()
regression = Thereefactor_regression(X,y)

price_checkbox = st.checkbox('Check the adjusted close price plot')
data_checkbox = st.checkbox('Check the merged data (monthly data)')
if price_checkbox:
    fig = go.Figure()
    for idx, col_name in enumerate(df):
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,idx]
                    ,name=df.columns[idx]
                    
                    ))
    fig.update_layout(height=500, width=800, title_text="Adj-close price")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Stock Price")
    st.plotly_chart(fig)

if data_checkbox:
    st.write(merge)
st.write("**Regression Summary Table**",regression.summary_table())

#
st.write("**Regression Models**")
for j in range(len(ASSETS),0,-1):
    st.write("**{}**".format(merge.columns[-j])+ " = " + " {:.4f} ".format(regression.alphas[-j+len(ASSETS)]) + "{0:+.4f} x **R_mkt**".format(regression.beta_1s[-j+len(ASSETS)])  + "  {0:+.4f} x **R_size**".format(regression.beta_2s[-j+len(ASSETS)])+ "  {0:+.4f} x **R_value**".format(regression.beta_3s[-j+len(ASSETS)]))
    
#st.write("**Detailed Regression Summary**")

#ticker = st.selectbox("Choose a Ticker for detailed regression summary",ASSETS)
#index = ASSETS.index(ticker)


#Y = merge.iloc[:,-(len(ASSETS)-index)]
#XX = sm.add_constant(X)
#model = sm.OLS(Y, XX)
#results = model.fit()

#st.table(results.summary())






