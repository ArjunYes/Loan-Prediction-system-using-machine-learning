from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from flask import send_file
#from imblearn.ensemble import BalancedBaggingClassifier
# importing library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
#import category_encoders as ce
#import pandas_profiling as pp


app = Flask(__name__,template_folder='templates')

def home():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])


def upload():
    if request.method == 'POST':
        lp_test = pd.read_csv(request.files.get('file'))

        lp_test2 = lp_test.copy()

        sub_inp=lp_test[['Unnamed: 0', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState',
       'CCSC', 'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 'NewExist',
       'CreateJob', 'RetainedJob', 'FranchiseCode', 'UrbanRural', 'RevLineCr',
       'LowDoc', 'ChgOffDate', 'DisbursementDate', 'DisbursementGross',
       'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']]

        lp_test['RevLineCr'].value_counts()
        lp_test['RevLineCr']=lp_test['RevLineCr'].map({'N':int('0'),'Y':int('1')})
        lp_test['RevLineCr'].value_counts()
        lp_test.info()
        RevLineCr_mode=lp_test['RevLineCr'].value_counts().index[0]
        RevLineCr_mode
        lp_test['RevLineCr'].fillna(RevLineCr_mode,inplace=True)
        lp_test.info()

        lp_test['LowDoc'].value_counts()
        lp_test['LowDoc']=lp_test['LowDoc'].str.replace('C','Nan')
        lp_test['LowDoc']=lp_test['LowDoc'].str.replace('1','Nan')
        lp_test['LowDoc'].value_counts()
        lp_test['LowDoc']=lp_test['LowDoc'].map({'N':int('0'),'Y':int('1')})
        lp_test['LowDoc'].value_counts()
        lp_test.info()
        lp_test['LowDoc'].value_counts()
        LowDoc_mode=lp_test['LowDoc'].value_counts().index[0]
        LowDoc_mode
        lp_test['LowDoc'].fillna(LowDoc_mode,inplace=True)
        lp_test.info()

        lp_test['GrAppv'].value_counts()
        lp_test['GrAppv']=lp_test['GrAppv'].str.replace('$',' ')
        lp_test['GrAppv']=lp_test['GrAppv'].str.replace(',','')
        lp_test['GrAppv']=lp_test['GrAppv'].astype(float)
        lp_test['GrAppv'].value_counts()
        lp_test.info()

        lp_test['SBA_Appv'].value_counts()
        lp_test['SBA_Appv']=lp_test['SBA_Appv'].str.replace('$',' ')
        lp_test['SBA_Appv']=lp_test['SBA_Appv'].str.replace(',','')
        lp_test['SBA_Appv']=lp_test['SBA_Appv'].astype(float)
        lp_test['SBA_Appv'].value_counts()

        lp_test['ChgOffPrinGr'].value_counts()
        lp_test['ChgOffPrinGr']=lp_test['ChgOffPrinGr'].str.replace('$',' ')
        lp_test['ChgOffPrinGr']=lp_test['ChgOffPrinGr'].str.replace(',','')
        lp_test['ChgOffPrinGr']=lp_test['ChgOffPrinGr'].astype(float)
        lp_test['ChgOffPrinGr'].value_counts()

        lp_test['BalanceGross'].value_counts()
        lp_test['BalanceGross']=lp_test['BalanceGross'].str.replace('$',' ')
        lp_test['BalanceGross']=lp_test['BalanceGross'].str.replace(',','')
        lp_test['BalanceGross']=lp_test['BalanceGross'].astype(float)
        lp_test['BalanceGross'].value_counts()

        lp_test['DisbursementGross'].value_counts()
        lp_test['DisbursementGross']=lp_test['DisbursementGross'].str.replace('$',' ')
        lp_test['DisbursementGross']=lp_test['DisbursementGross'].str.replace(',','')
        lp_test['DisbursementGross']=lp_test['DisbursementGross'].astype(float)
        lp_test['DisbursementGross'].value_counts()
        DG_mode=lp_test['DisbursementDate'].value_counts().index[0]
        DG_mode
        lp_test['DisbursementDate'].fillna(DG_mode,inplace=True)
        lp_test.info()

        # Create correlation matrix
        corr_matrix = lp_test.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        lp_test.drop(lp_test[to_drop], axis=1,inplace=True)

        # Outlier of NoEmp
        #it is highly Skewed.removing the outliers and imputating it.

        ## Perform all the steps of IQR
        sorted(lp_test['NoEmp'])
        quantile1, quantile3= np.percentile(lp_test['NoEmp'],[25,75])
        print(quantile1,quantile3)
        ## Find the IQR
        iqr_value=quantile3-quantile1
        print(iqr_value)
        ## Find the lower bound value and the higher bound value
        lower_bound_val = quantile1 -(1.5 * iqr_value) 
        upper_bound_val = quantile3 +(1.5 * iqr_value)
        print(lower_bound_val,upper_bound_val)
        lp_test['NoEmp'] = np.where(lp_test['NoEmp'] <-7.0, -7.0,lp_test['NoEmp'])
        lp_test['NoEmp'] = np.where(lp_test['NoEmp'] >17.0, 17.0,lp_test['NoEmp'])

        # Outlier of RetainedJob
        #it is highly Skewed.removing the outliers and imputating it.
        ## Perform all the steps of IQR
        sorted(lp_test['RetainedJob'])
        quantile1, quantile3= np.percentile(lp_test['RetainedJob'],[25,75])
        print(quantile1,quantile3)
        ## Find the IQR
        iqr_value=quantile3-quantile1
        print(iqr_value)
        ## Find the lower bound value and the higher bound value
        lower_bound_val = quantile1 -(1.5 * iqr_value) 
        upper_bound_val = quantile3 +(1.5 * iqr_value)
        print(lower_bound_val,upper_bound_val)
        lp_test['RetainedJob'] = np.where(lp_test['RetainedJob'] <-6.0, -6.0,lp_test['RetainedJob'])
        lp_test['RetainedJob'] = np.where(lp_test['RetainedJob'] >10.0, 10.0,lp_test['RetainedJob'])
        #RetainedJob variable having zeroes. using imputation replacing the zero value

        imp = SimpleImputer(missing_values=0,strategy = 'most_frequent')

        mode_imp = SimpleImputer(strategy= 'most_frequent')

        dfseries = lp_test.RetainedJob

        dfseries.values
        dfseries.values.reshape(-1,1).shape
        reshdf = dfseries.values.reshape(-1,1)

        reshdf.shape
        imp.fit_transform(reshdf)
        lp_test.RetainedJob = imp.fit_transform(reshdf)
        lp_test['RetainedJob'].describe()

                #it is highly Skewed.removing the outliers and imputating it.
        ## Perform all the steps of IQR
        sorted(lp_test['CreateJob'])
        quantile1, quantile3= np.percentile(lp_test['CreateJob'],[25,75])
        print(quantile1,quantile3)
        ## Find the IQR
        iqr_value=quantile3-quantile1
        print(iqr_value)
        ## Find the lower bound value and the higher bound value
        lower_bound_val = quantile1 -(1.5 * iqr_value) 
        upper_bound_val = quantile3 +(1.5 * iqr_value)
        print(lower_bound_val,upper_bound_val)

        #New Exist variable having zeroes. using imputation replacing the zero value

        imp = SimpleImputer(missing_values=0,strategy = 'most_frequent')

        mode_imp = SimpleImputer(strategy= 'most_frequent')

        dfseries = lp_test.NewExist

        dfseries.values
        dfseries.values.reshape(-1,1).shape
        reshdf = dfseries.values.reshape(-1,1)

        reshdf.shape
        imp.fit_transform(reshdf)
        lp_test.NewExist = imp.fit_transform(reshdf)
        lp_test['NewExist'].describe()

        Name_mode=lp_test['Name'].value_counts().index[0]
        Name_mode
        lp_test['Name'].fillna(Name_mode,inplace=True)
        lp_test.info()

        State_mode=lp_test['State'].value_counts().index[0]
        State_mode
        lp_test['State'].fillna(State_mode,inplace=True)
        lp_test.info()

        BankState_mode=lp_test['BankState'].value_counts().index[0]
        BankState_mode
        lp_test['BankState'].fillna(BankState_mode,inplace=True)
        lp_test.info()

        Bank_mode=lp_test['Bank'].value_counts().index[0]
        Bank_mode
        lp_test['Bank'].fillna(Bank_mode,inplace=True)
        lp_test.info()

        #Intuitively insignificant variables are deleted.
        lp_test.drop(['Unnamed: 0','Name','ChgOffDate','City','BankState','Bank','State','CCSC','FranchiseCode','ChgOffPrinGr'],axis=1,inplace=True)


        label_test_enc_dict={}
        lp_test_clean_enc=lp_test.copy()

        for col_name in lp_test_clean_enc:

            label_test_enc_dict[col_name]=LabelEncoder()

            col=lp_test_clean_enc[col_name]
            col_not_null=col[col.notnull()]
            reshaped_vals=col_not_null.values.reshape(-1,1)


            encoded_vals=label_test_enc_dict[col_name].fit_transform(reshaped_vals)

            lp_test_clean_enc.loc[col.notnull(),col_name]=np.squeeze(encoded_vals)




        model = pickle.load(open("model.pkl", "rb"))
        result1= model.predict( lp_test_clean_enc) 
        ss1 = pd.DataFrame(result1, columns=[ 'prediction'])
        ss1.loc[ss1['prediction']== 1,'prediction']="CHGOFF"
        ss1.loc[ss1['prediction']== 0,'prediction']="PIF"
   
        output = pd.concat([lp_test2, ss1],axis=1)
        a = np.random.randn()
        output.to_csv(r'output'+str(a)+'.csv',index=False)
        
        return render_template('upload.html', result=ss1)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)