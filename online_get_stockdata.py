import pandas as pd
import numpy as np
import ssl
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
from pandas.util.testing import assert_frame_equal
import datetime as dt
import pandas_datareader.data as web


#must read https://kopu.chat/2017/05/28/%E5%8D%81%E5%88%86%E9%90%98%E8%AE%80%E6%87%82%E7%90%86%E8%B2%A1%E6%8A%95%E8%B3%87%E5%AD%B8-%E4%BB%80%E9%BA%BC%E6%98%AF%E6%9C%80%E5%A5%BD%E7%9A%84%E6%8A%95%E8%B3%87%E7%B5%84%E5%90%88%EF%BC%9F/
# package to extract data from various Internet sources into a DataFrame
# make sure you have it installed


ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


#Create your time range (start and end variables).
# The start date should be 01/01/2015 and the end should today (whatever your today is)

start = dt.datetime(2015,1,1)

end= dt.datetime.today()

print(start)


stocks=['AAPL','TSLA','IBM']

df = web.DataReader(stocks, 'yahoo', start, end)
print(df.head())

print(df.index)
print(df.columns)

print(df.shape)

print(df.info())

print(df.describe())

#Calculate the daily returns of the adjusted close price
df['Returns'] = StockPrices['Adjusted'].pct_change()
# Convert the decimal returns into percentage returns
percent_return = df['Returns']*100

df['Adj Close'].plot(legend=True, figsize=(10,4))
plt.show()


#############################
##Annualizing mean//variance#
#############################


# You can use the formula below to estimate the average annual return of a stock
#given the average daily return and the number of trading days in a year
#(typically there are roughly 252 trading days in a year):

#Average Annualized Return=((1+μ)**252)−1


import numpy as np

# Calculate the average daily return of the stock
mean_return_daily = np.mean(df['Returns'])
print(mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print(mean_return_annualized)

sigma_daily = np.std(df['Returns'])
print(sigma_daily)

# Calculate the daily variance
variance_daily = sigma_daily**2
print(variance_daily)

sigma_annualized = sigma_daily*np.sqrt(252)
print(sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print(variance_annualized)

########################
##skewness$$Kurtosis####
########################

# Import skew from scipy.stats
from scipy.stats import skew

# Drop the missing values
clean_returns = df['Returns'].dropna()

# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print(returns_skewness)

# Import kurtosis from scipy.stats
from scipy.stats import kurtosis

# Calculate the excess kurtosis of the returns distribution
excess_kurtosis = kurtosis(StockPrices['Returns'].dropna())
print(excess_kurtosis)

# Derive the true fourth moment of the returns distribution
fourth_moment = excess_kurtosis+3
print(fourth_moment)


########################
##Normality test########
########################


# Import shapiro from scipy.stats
from scipy.stats import shapiro

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(StockPrices['Returns'].dropna())
print("Shapiro results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)

if p_value <= 0.05:
print("Null hypothesis of normality is rejected.")
else:
print("Null hypothesis of normality is accepted.")



#######################
##porffolio returns####
#######################

# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
CumulativeReturns.plot()
plt.show()



#Equal weighted portfolios

# How many stocks are in your portfolio?
numstocks = 9

# Create an array of equal weights across all assets
portfolio_weights_ew = np.repeat(1/numstocks,numstocks)

# Calculate the equally-weighted portfolio returns
StockReturns['Portfolio_EW'] = StockReturns.iloc[:, 0:numstocks].mul(portfolio_weights_ew, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW']

#给予每一个证券的权重等于该证券市值在整个组合市值中的权重决定。
#市值加权指数假设投资者按照每只股票的市值比例去投资。

# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])  

# Calculate the market cap weights
mcap_weights = market_capitalizations/sum(market_capitalizations)

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MCap'])



######################
#Corr&COV&&heatmap####
#####################                      

# Import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = StockReturns.corr()

# Print the correlation matrix
print(correlation_matrix)
                        
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()

# Calculate the covariance matrix
cov_mat = StockReturns.cov()

# Annualize the co-variance matrix
cov_mat_annual = cov_mat*252

# Print the annualized co-variance matrix
print(cov_mat_annual)


#Sharpe ratios

# Risk free rate
risk_free = 0

# Calculate the Sharpe Ratio for each asset
RandomPortfolios['Sharpe'] = (RandomPortfolios['Returns'] - risk_free) / RandomPortfolios['Volatility']

# Print the range of Sharpe ratios
print(RandomPortfolios['Sharpe'].describe()[['min', 'max']])

#Portfolio standard deviation
# Import numpy as np
import numpy as np
                        
#MSR MAX SHARPE ROTIO                        
# Calculate the portfolio standard deviation
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print(portfolio_volatility)


# Sort the portfolios by Sharpe ratio
sorted_portfolios = RandomPortfolios.sort_values(by=['Sharpe'], ascending=False)

# Extract the corresponding weights
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the MSR weights as a numpy array
MSR_weights_array = np.array(MSR_weights)

# Calculate the MSR portfolio returns
StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)
                        
#####################################
#GMV GLOBAL MINI VOLATILITY PORFOLIO#
#####################################
                        
# Sort the portfolios by volatility
sorted_portfolios = RandomPortfolios.sort_values(by=['Volatility'], ascending=True)

# Extract the corresponding weights
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the GMV weights as a numpy array
GMV_weights_array = np.array(GMV_weights)

# Calculate the GMV portfolio returns
StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR', 'Portfolio_GMV'])

######################
#CAPM#################
######################

# Calculate the co-variance matrix between Portfolio_Excess and Market_Excess
covariance_matrix = FamaFrenchData[['Portfolio_Excess', 'Market_Excess']].cov()

# Extract the co-variance co-efficient
covariance_coefficient = covariance_matrix.iloc[0, 1]
print(covariance_coefficient)

# Calculate the benchmark variance
benchmark_variance = FamaFrenchData['Market_Excess'].var()
print(benchmark_variance)

# Calculating the portfolio market beta
portfolio_beta = covariance_coefficient/benchmark_variance
print(portfolio_beta)
                        
