import pandas as pd
import numpy as np

sl_df = pd.read_excel("../Data/Copy of Service Line Data.xls")
res_df = pd.read_csv("../Data/residential_test_data.csv")
test_df = pd.read_csv("../Data/parcel_geo_data_with_SL_types.csv")

id_test = test_df['PID Dash']
test_df_old = test_df
merged_df = pd.merge(res_df, sl_df, left_on='PID Dash', right_on='PIDdash',how='left')
ytrue = merged_df['Lead (ppb)']

merged_df.rename(columns={'Latitude_x': 'Latitude', 'Longitude_x': 'Longitude'}, inplace=True)
test_df.rename(columns={'Street_Name': 'Street Name'}, inplace=True)

Ydata = (merged_df['Lead (ppb)'] > 15)

#merged_df['Property Address']
#test_df['Property Address']

#cols_to_keep = ['Street #', 'Street Name', 'Property Zip Code', 'PID no Dash', 'Owner Type', 'Homestead', 'Homestead Percent', 'HomeSEV', 'Land Value', 'Land Improvements Value', 'Residential Building Value', 'Residential Building Style', 'Commercial Building Value', 'Building Storeys', 'Parcel Acres', 'Rental', 'Use Type', 'Old Prop class', 'USPS Vacancy', 'Zoning', 'Future Landuse', 'DRAFT Zone', 'Housing Condition 2012', 'Housing Condition 2014', 'Commercial Condition 2013', 'Latitude', 'Longitude', 'Hydrant Type', 'Ward', 'PRECINCT', 'CENTRACT', 'CENBLOCK', 'Acres', 'Year_Built', 'Year Built', 'SL_Type', 'SL_Type2', 'SL_Lead', 'Class']

#cols_to_factorize = ['Street Name','Zip Code','Property Zip Code','Owner Type','Homestead',
#                    'Residential Building Style','Rental','Use Type','Old Prop class',
#                    'USPS Vacancy','Zoning','Future Landuse','DRAFT Zone','Housing Condition 2012',
#                    'Housing Condition 2014','Commercial Condition 2013','Hydrant Type','Class',
#                    'Street #']

cols_to_keep = ['Property Zip Code', 'PID no Dash', 'Owner Type', 'Homestead', 'Homestead Percent', 'HomeSEV', 'Land Value', 'Land Improvements Value', 'Residential Building Value', 'Residential Building Style', 'Commercial Building Value', 'Building Storeys', 'Parcel Acres', 'Rental', 'Use Type', 'Old Prop class', 'USPS Vacancy', 'Zoning', 'Future Landuse', 'DRAFT Zone', 'Housing Condition 2012', 'Housing Condition 2014', 'Commercial Condition 2013', 'Latitude', 'Longitude', 'Hydrant Type', 'Ward', 'PRECINCT', 'CENTRACT', 'CENBLOCK', 'Acres', 'Year_Built', 'Year Built', 'SL_Type', 'SL_Type2', 'SL_Lead', 'Class']#,'Street Name']

cols_to_factorize = ['Zip Code','Property Zip Code','Owner Type','Homestead',
                     'Residential Building Style','Rental','Use Type','Old Prop class',
                     'USPS Vacancy','Zoning','Future Landuse','DRAFT Zone','Housing Condition 2012',
                     'Housing Condition 2014','Commercial Condition 2013','Hydrant Type','Class']#,'Street Name']

cols_to_bayesian = ['Property Zip Code', 'Building Storeys', 'Use Type', 'Old Prop class', 'USPS Vacancy', 'Zoning', 'Future Landuse', 'DRAFT Zone', 'Housing Condition 2012', 'Housing Condition 2014', 'Commercial Condition 2013', 'Hydrant Type', 'Class', 'Residential Building Style', 'Ward', 'PRECINCT', 'CENTRACT', 'CENBLOCK', 'res_year_built', 'sl_year_built', 'SL_Type', 'SL_Type2', 'SL_Lead', 'Homestead', 'Homestead Percent', 'Rental']
#cols_to_bayesian = ['SL_Type', 'SL_Lead']
#cols_to_bayesian = ['Property Zip Code', 'Building Storeys', 'Use Type', 'Old Prop class', 'USPS Vacancy', 'Zoning', 'Future Landuse', 'DRAFT Zone', 'Hydrant Type', 'Class', 'SL_Type'] 

merged_df = merged_df[cols_to_keep]
test_df = test_df[cols_to_keep]

for i in cols_to_keep:
    print i, np.unique(merged_df.loc[merged_df[i].notnull(), i]).shape

all_df = pd.concat([merged_df, test_df], axis = 0)

#all_df['Ward'].replace(to_replace=' ', value='', inplace=True)
#all_df['PRECINCT'].replace(to_replace=' ', value='', inplace=True)

all_df.rename(columns={'Year Built': 'res_year_built', 'Year_Built': 'sl_year_built'}, inplace=True)

for i, row in all_df.iterrows():
    row['Property Zip Code'] = str(row['Property Zip Code'])[:5]

# this doesn't work right now
all_df.replace(to_replace=" ",value="",inplace=True)
all_df.fillna(value=-999,inplace=True)

# factorize string-based features for XGBoost
for (feat_name, feat_series) in all_df.iteritems():
    if feat_name in cols_to_factorize:
        all_df[feat_name], tmp_indexer = pd.factorize(all_df[feat_name])
    print(feat_name)
    all_df[feat_name] = all_df[feat_name].astype(float)

# Now that our data is loaded, clean, and ready to go, create numpy arrays to pass                                                                                     
# to a classifier.                                                                                                                                                      

Xdata = all_df[0:merged_df.shape[0]]
Xtest = all_df[merged_df.shape[0]:]

ratio_threshold = 30

for col in cols_to_bayesian:
    size0 = pd.concat([Xdata[col], Ydata], axis = 1)
    size1 = size0.loc[size0['Lead (ppb)'] == 1].groupby([col]).count().reset_index()
    size1.columns = [col, 'subcount']
    size2 = pd.concat([Xdata[col], Ydata], axis = 1).groupby([col]).count().reset_index()
    size2.columns = [col, 'count']
    size3 = pd.merge(size1, size2, how = 'right', on = [col])
    size3.fillna(value=0, inplace=True)
    size3[col + ' positive ratio'] = size3['subcount'] / size3['count']
    print size3
    size3.loc[size3['count'] < ratio_threshold, col + ' positive ratio'] = -999
    size3 = size3[[col, col + ' positive ratio']]
    Xdata = pd.merge(Xdata, size3, how = 'left')
    Xtest = pd.merge(Xtest, size3, how = 'left')

Xdata.fillna(value=-999, inplace=True)
Xtest.fillna(value=-999, inplace=True)

#print all_df

print Xdata.columns

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import roc_auc_score as roc
from sklearn.cross_validation import train_test_split

class XGBClassifier_new(XGBClassifier):
    def predict(self, X):
        return XGBClassifier.predict_proba(self, X)[:,1]

xgtrain = xgb.DMatrix(Xdata,label=Ydata)
xgmodel = XGBClassifier_new(n_estimators=550,seed=1,learning_rate=0.1,objective='binary:logistic',nthread=-1)
xgb_param = xgmodel.get_xgb_params()
cvresult = xgb.cv(xgb_param,xgtrain, num_boost_round=xgmodel.get_params()['n_estimators'],nfold=4,metrics=['auc'],
                 early_stopping_rounds=100, show_progress=True)

#cross_validation = StratifiedKFold(Ydata, n_folds=4, shuffle=True,random_state=0)
#predicted = cross_val_predict(xgmodel, Xdata, Ydata, cv = cross_validation, verbose= 1, n_jobs = 1)

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.25, random_state=1)

xgmodel.fit(X_train, y_train)
ypred = xgmodel.predict_proba(X_test)

print roc(y_test, ypred[:,1])

#print ypred[:,1]

xgmodel.fit(Xdata, Ydata)

test_df_old['predicted'] = xgmodel.predict_proba(Xtest)[:,1]

#print test_df_old

test_df_old.to_csv('submission_xgb.csv',index=False)

merged_df['predicted'] = xgmodel.predict_proba(Xdata)[:,1]
merged_df['Ytrue'] = ytrue

#print roc(ytrue, merged_df['predicted'])

merged_df.to_csv('train_xgb.csv',index=False)

#import matplotlib.pyplot as plt

#feature_importance = pd.Series(xgmodel.booster().get_fscore()).sort_values(ascending=False)
#print feature_importance
#feature_importance.plot(kind='bar', title='Feature Importances')
#plt.ylabel('Feature Importance Score')
#plt.show()

#feature_importance.to_csv("feature_importance.csv", index = True)

exit()

from sklearn.metrics import roc_curve

fig = plt.figure()
fig.set_size_inches(8,8)

fpr, tpr, _ = roc_curve(Ytest, yhat1[:,1])
plt.plot(fpr, tpr, label= 'LogReg (area = %0.5f)' % r1)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Lead Classifiers')
plt.legend(loc="lower right")

plt.show()
