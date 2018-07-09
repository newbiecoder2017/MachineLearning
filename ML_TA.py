import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

#read the raw data file
# ta_df = pd.read_csv("c:/Python27/Git/MachineLearning/indicators.csv", index_col = 'Date', parse_dates=True)

def regression_model(m):
    ta_df = pd.read_csv("C:/Python27/Git/MachineLearning/indicators.csv", index_col='Date', parse_dates=True)
    # calculate the price change and shift it by n period forward
    period_fwd = m
    ta_df['price_change'] = ta_df.close.pct_change().shift(-period_fwd)

    #dataframe column_names
    cols = ta_df.columns.tolist()
    cols.remove('close')

    #dataframe with no close
    ta_df = ta_df[cols]

    #calculate the dataframe correlation matrix
    corr_mat = ta_df.corr()

    #calculate the correlation with the fwd price change only
    cmat_with_price_change = corr_mat['price_change'].abs().sort_values(ascending = False)

    #get the the dependent variable names from the corr mat
    target = cmat_with_price_change.index[0]

    #select only the top N highly correlated predictors
    predictors = cmat_with_price_change.index[1:8]

    #list of predictors names
    req_cols = cmat_with_price_change.index[1:8]

    #create a df with selected predictors and target
    reg_df = ta_df[req_cols].copy()
    reg_df[target] = ta_df[target]
    reg_df = reg_df[:-period_fwd]


    #histogram plot
    # reg_df.hist()
    # plt.show()

    #boxplot
    # reg_df.boxplot()
    # plt.show()

    #normalize the regr dataframe
    mmscale = preprocessing.MinMaxScaler().fit(reg_df)
    norm_df = pd.DataFrame(mmscale.transform(reg_df), columns=reg_df.columns)

    #box plot of norm df
    # norm_df.boxplot()
    # plt.show()

    #remove_outliers from the normalized df
    def remove_outlier(df):
        low = 0.02
        high = 0.96
        quantile_df = df.quantile([low,high])
        for name in list(df.columns):
            df = df[(df[name]>quantile_df.loc[low, name]) & (df[name]<quantile_df.loc[high, name])]
        return df

    preproc_df = remove_outlier(norm_df)

    #plot preproc df box plot
    # preproc_df.boxplot()
    # # plt.show()

    #spliting the data into training set and tetsing set
    train_set, test_set = train_test_split(preproc_df, test_size=0.3)

    #assign X and Y for trainings set
    # train_set_x = train_set[train_set.columns[1:6]]
    train_set_x = train_set[req_cols]

    # train_set_y = train_set[train_set.columns[0,np.newaxis]]
    train_set_y = train_set['price_change']

    #assign X and Y for testing set
    # test_set_x = test_set[test_set.columns[1:6]]
    test_set_x = test_set[req_cols]

    # test_set_y = test_set[test_set.columns[0,np.newaxis]]
    test_set_y = test_set['price_change']

    # print the shape of test and train set
    # print(train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)

    #fitting Liner Model
    lin_mod = linear_model.LinearRegression()
    lin_mod.fit(train_set_x, train_set_y)
    print("coef for: " + str(m) + " is ", lin_mod.coef_)
    #predict using the fitted model
    test_predict_y = lin_mod.predict(test_set_x)
    # plt.plot(test_set_y, test_predict_y)
    # plt.show()
    #evaluate the accuracy of the fitted model
    mse = mean_squared_error(test_set_y, test_predict_y)
    rsquare = r2_score(test_set_y,test_predict_y)
    return mse, rsquare
    #convert the scale data into original scale
    # mse_original = mmscale.inverse_transform(train_set)

    #plot the predicted value and original value
    # plt.scatter(test_set_y, test_predict_y,c='rg')
    # plt.legend()
    # plt.show()

mse_ls = []
rsq_ls = []
per_fwd = [1,5,10,15,20,25,30,60,90]
for i in per_fwd:
    msee, rsq = regression_model(i)
    mse_ls.append(msee)
    rsq_ls.append(rsq)

plot_df = pd.DataFrame({'MSE':mse_ls, 'Rsq':rsq_ls}, index = per_fwd)
plot_df.to_csv("C:/Python27/Git/MachineLearning/liner_regression_output.csv")
print(plot_df)
plot_df['Rsq'].plot()
plt.legend()
plt.show()