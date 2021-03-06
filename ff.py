import pandas as pd
import datetime as dt
import statsmodels.api as sm
import streamlit as st
import numpy as np
from datetime import timedelta
import yfinance as yf
from imputs import  get_date,get_list
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt






ASSETS = get_list()# get variables
ASSETS = [x for x in ASSETS if x != '']#remove blank value
ASSETS = list(dict.fromkeys(ASSETS))#remove duplicated value
ASSETS.sort()
month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

st.title("Fama French 3 Factors Analysis")
'''#### (stock prices from *yahoo finance*) '''
st.subheader("**Current Stock List**")
st.write(list(dict.fromkeys(ASSETS)))

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
        self.pvalues = []
        X = sm.add_constant(X)
        for ind, col in enumerate(y.columns):
            a = y.iloc[:,ind]
            model = sm.OLS(a, X)
            results = model.fit()
            alpha, beta_1, beta_2, beta_3 = results.params
            t = results.tvalues
            p = results.pvalues
            r2_adj = results.rsquared_adj
            r2 = results.rsquared
            mse = results.mse_resid
            conf_int = results.conf_int()
            resid = results.resid
            self.pvalues.append(p)
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
        self.alpha_pvalues = [self.pvalues[i][0] for i in range(len(y.columns))]
        self.beta_1_pvalues = [self.pvalues[i][1] for i in range(len(y.columns))]
        self.beta_2_pvalues = [self.pvalues[i][2] for i in range(len(y.columns))]
        self.beta_3_pvalues = [self.pvalues[i][3] for i in range(len(y.columns))]
        self.resids = pd.DataFrame(self.resids,index = y.columns).T #creat a residual dataframe
        self.sd_resid = (self.resids - self.resids.mean())/self.resids.std() #calculating the standize residual
    def summary_table(self):
        table = pd.DataFrame({"??":self.alphas
                            ,"95% CI for ??":self.alpha_CI
                            ,"?? pvalues":self.alpha_pvalues
                            ,"??1":self.beta_1s
                            ,"95% CI for ??1":self.beta_1_CI
                            ,"??1 pvalues":self.beta_1_pvalues
                            ,"??2":self.beta_2s
                            ,"95% CI for ??2":self.beta_2_CI
                            ,"??2 pvalues":self.beta_2_pvalues
                            ,"??3":self.beta_3s
                            ,"95% CI for ??3":self.beta_3_CI
                            ,"??3 pvalues":self.beta_3_pvalues
                            ,"R2": self.r2s
                            ,"R2_adj":self.r2_adjs
                            ,"SER":np.sqrt(self.mses)
                            },index = self.y.columns)
        return table
######
def siginificance(x,sig_level = 0.05):
    c1 = 'background-color: yellow'
    c2 = '' 
    #compare columns
    alpha_mask = x['?? pvalues'] < sig_level
    #DataFrame with same index and columns names as original filled empty strings
    df1 =  pd.DataFrame(c2, index=x.index, columns=x.columns)
    #modify values of df1 column by boolean mask
    df1.loc[alpha_mask, '??'] = c1
    
    beta_1_mask = x['??1 pvalues'] < sig_level
    df1.loc[beta_1_mask, '??1'] = c1
    
    beta_2_mask = x['??2 pvalues'] < sig_level
    df1.loc[beta_2_mask, '??2'] = c1
    
    beta_3_mask = x['??3 pvalues'] < sig_level
    df1.loc[beta_3_mask, '??3'] = c1
    return df1
def negative_red(val):
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color
######


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
    st.write("Data starting date",START_DATE,"Data ending date",END_DATE)

if data_checkbox:
    st.write(merge)
    st.write("Shape of data: ", merge.shape)
    

summary_df = regression.summary_table()

#for efficiency consideration, close out it.

#@st.cache
#def Summary_table():
#    sector_name = []
#    for comp in ASSETS:
#      sector = yf.Ticker(comp)
#      sector = sector.info["sector"]
#       sector_name.append(sector)
#    industry = pd.DataFrame({"Industry": sector_name}, index = summary_df.index)
#    return industry
#industry = Summary_table()

#summary_df = pd.merge(industry,summary_df,left_index=True, right_index=True)




st.write("")
st.write("")
st.subheader("**Regression Summary Table**")
sig = st.select_slider('Slide to select the significance level', options=['0.01','0.05','0.1'])
#summary_df = summary_df.style.apply(siginificance,sig_level = float(sig), axis=None).applymap(negative_red,subset=['??',"??1","??2","??3"])
st.write(summary_df)

#
st.write("The cells highlight in yellow are significant in {} significance level.".format(sig))
st.write("")
model_checkbox = st.checkbox('Check the regression model for each stock.')
if model_checkbox:
    st.subheader("**Regression Models**")
    for j in range(len(ASSETS),0,-1):
        st.write("**{}**".format(merge.columns[-j])+ " = " + " {:.4f} ".format(regression.alphas[-j+len(ASSETS)]) + "{0:+.4f} x **(MKT-RF)**".format(regression.beta_1s[-j+len(ASSETS)])  + "  {0:+.4f} x **SMB**".format(regression.beta_2s[-j+len(ASSETS)])+ "  {0:+.4f} x **HML**".format(regression.beta_3s[-j+len(ASSETS)]) + "+ ??")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
st.subheader("**Detailed Regression Summary**")

ticker = st.selectbox("Choose a Ticker for detailed regression summary",ASSETS)

try:
    tickerData = yf.Ticker(ticker)
    #logo
    string_logo = '<img src=%s>' % tickerData.info['logo_url']
    st.markdown(string_logo, unsafe_allow_html=True)
    #
    string_name = tickerData.info["sector"]
    st.subheader('**%s**' % string_name)
except:
	pass
	


index = ASSETS.index(ticker)
Y = merge.iloc[:,-(len(ASSETS)-index)]
XX = sm.add_constant(X)
model = sm.OLS(Y, XX)
results = model.fit()

corre = pd.merge(Y,X,left_index=True, right_index=True)
fig = plt.figure(figsize=(13, 6))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(corre.corr(), vmin=-1, vmax=1, annot=True,annot_kws={"size": 14})
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':16}, pad=12)
#rho, pval = pearson(corre)
#gram = corrgram(rho, pval, corre.columns, dpi=120)
st.pyplot(fig)

###
results_summary1 = results.summary2().tables[0]
results_summary1 = results_summary1.assign(hack='').set_index('hack')
results_summary1.columns = results_summary1.iloc[0]
results_summary1 = results_summary1[1:]
####
results_summary2 = results.summary2().tables[1]
results_summary3 = results.summary2().tables[2]
results_summary3 = results_summary3.assign(hack='').set_index('hack')
results_summary3.columns = results_summary3.iloc[0]
results_summary3 = results_summary3[1:]

st.table(results_summary1)
st.table(results_summary2)
st.table(results_summary3)

st.write("")
st.write("")
st.write("**check out this for the codes in this app** [link](https://github.com/Slin-Miraka/Fama-French/)")
st.write("**check out this for Mean-Variance Framework app** [link](https://share.streamlit.io/slin-miraka/efficient-frontier-app/main/MVF.py)")
email = "slin3137@uni.sydney.edu.au"
st.write("**My E-mail**: ", email)


#Y = merge.iloc[:,-(len(ASSETS)-index)]
#XX = sm.add_constant(X)
#model = sm.OLS(Y, XX)
#results = model.fit()

#st.table(results.summary())






