import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore

def shapiro_test(df):
    W, p = shapiro(df.dropna())
    if p < 0.05:
        print(f"{df.name}: Probably Gaussian (W = {W}, p = {p}).")
    else:
        print(f"{df.name}: Probably not Gaussian (W = {W}, p = {p}).")
    return W, p

def find_outliers_Zscore(df, alpha=3, plot=False):
    df['z'] = zscore(df, nan_policy='omit')
    outliers = df[(df.z<-alpha) | (df.z>alpha)]
    mu = df.iloc[:, 0].mean()
    std = df.iloc[:, 0].std()

    if plot is True:
        plt.scatter(df.index.values, df.iloc[:, 0])
        plt.scatter(outliers.index.values, outliers.iloc[:, 0], c='r')
        plt.axhline(y=mu, color='k', linestyle='--')
        plt.axhline(y=mu+alpha*std, color='grey', linestyle='-')
        plt.axhline(y=mu-alpha*std, color='grey', linestyle='-')
        plt.title(df.columns[0])
        plt.show()
    return outliers

def find_outliers_discrete(df, alpha=3, plot=False, verbose=False):
    freq = df.value_counts()
    support = np.sort(freq.index.values)
    mu = np.mean(support)
    std = np.std(support)
    min_thrd = mu - alpha * std
    max_thrd = mu + alpha * std
    outliers = df[((df < min_thrd) | (df > max_thrd))]
    
    if verbose is True:
        print(f"Found {len(outliers)} outliers for {df.name}.")

    if plot is True:
        plt.scatter(df.index, df)
        plt.scatter(outliers.index, outliers, c='r')
        plt.axhline(y=mu, color='k', linestyle='--')
        plt.axhline(y=min_thrd, color='grey', linestyle='-')
        plt.axhline(y=max_thrd, color='grey', linestyle='-')
        plt.title(df.name)
        plt.show()
    return outliers

def find_outliers_IQR(df, k=1.5, plot=False):
    stats = df.describe()
    stats.loc['mode'] = float(df.mode().tolist()[0])
    # stats.loc['var'] = df.var()
    # stats.loc['skew'] = df.skew().tolist()
    # stats.loc['kurt'] = df.kurtosis().tolist()
    # print(stats)

    mu = df.mean()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    min_thrd = q1 - k * IQR
    max_thrd = q3 + k * IQR
    outliers = df[((df < min_thrd) | (df > max_thrd))]
    print(f"Found {len(outliers)} outliers for {df.name}.")

    if plot is True:
        plt.scatter(df.index, df)
        plt.scatter(outliers.index, outliers, c='r')
        plt.axhline(y=mu, color='k', linestyle='--')
        plt.axhline(y=min_thrd, color='grey', linestyle='-')
        plt.axhline(y=max_thrd, color='grey', linestyle='-')
        plt.show()
    return outliers

# from sklearn.ensemble import IsolationForest
# def find_outliers_IF(df):
#     clf = IsolationForest(max_samples=100, random_state=1, contamination='auto')
#     df['pred'] = clf.fit_predict(df).tolist()
#     outliers = df[df.pred == -1]
#     return outliers

# def rensac_regr():
#     ransac = RANSACRegressor(random_state=42).fit(X, y)
#     fit_df["ransac_regression"] = ransac.predict(plotline_X)
#     ransac_coef = ransac.estimator_.coef_
#     coef_list.append(["ransac_regression", ransac.estimator_.coef_[0]])

