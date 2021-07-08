# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:56:57 2020

@author: dell
"""
"using linear SVM to get feature coef"




import numpy as np
from sklearn import svm  
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
import pickle




def my_svm(x_train,y_train,x_test,y_test):
    model = svm.SVC(kernel='linear',probability=True)  # linear-kernal, probability=True
    
    # GridSearchCV for optimal parameters, 3-fold CV
    c_range = np.logspace(-15,-5, 15, base=2)
    param_grid = [{'kernel': ['linear'], 'C': c_range}]    
    gridsearch = GridSearchCV(model, param_grid, cv=3)
    
    # training 
    gridsearch.fit(x_train, y_train)

    # choose the best model
    best_model=gridsearch.best_estimator_
    # testing 
    score_test = best_model.score(x_test, y_test)
    score_train = best_model.score(x_train, y_train)
    y_pred=best_model.predict(x_test)
    y_prob=best_model.predict_proba(x_test)
    y_df = best_model.decision_function(x_test)
    print('The best parameters are %s, acc_train is %s, acc_test is %s' % (gridsearch.best_params_,score_train,score_test))
    feature_wei=best_model.coef_
    return y_pred,y_prob,y_df,feature_wei,best_model
 


def mci_test(x_mci,model):
    y_prob=model.predict_proba(x_mci)
    y_df=model.decision_function(x_mci)
    return y_prob,y_df

   

def my_model_performance(y_true,y_pred):
    sen=recall_score(y_true,y_pred,pos_label=3)
    spe=recall_score(y_true,y_pred,pos_label=1)
    acc=accuracy_score(y_true,y_pred)
    return acc,sen,spe

    

def feature_generation(smooth_size) :
    file = 'MCAD_raw_AFQ.mat'
    data = loadmat(file, mat_dtype=True)
    alldata=data['MCAD_AFQ_data'];
    ll=['FA','MD','RD','AD','CL','VOLUME']
    smooth=smooth_size
    out=np.empty(shape=(825,0))
    for j in range(0,len(ll)):
        print('processing measure:',ll[j])
        for i in range(0,20):
            print('fiber %s'%(i+1))
            a=alldata[i][0][ll[j]]
            for k in range(0,int(100/smooth)):
                temp1=a[:,k*smooth:(k+1)*smooth]
                temp2=np.nanmean(temp1,1)
                out=np.hstack((out,temp2.reshape(825,1)))
    return out



def combat_feature(smooth_size):
    file = 'combat_all12000.mat'
    data = loadmat(file, mat_dtype=True)
    alldata=data['feature']
    smooth=smooth_size
    out=np.empty(shape=(825,0))
    for i in range(0,int(12000/smooth)):
        temp1=alldata[:,i*smooth:(i+1)*smooth]
        temp2=np.nanmean(temp1,1)
        out=np.hstack((out,temp2.reshape(825,1)))
    return out




if __name__ == '__main__' :
    file = 'demography.mat'
    data = loadmat(file, mat_dtype=True)
    center=data['center']
    label=data['label']   ### 1-NC, 2-MC, 3-AD
    popu=data['popu']
    
    ##### set average-smooth kernel ####
    sss=5
    
    ##### choose original data or harmonized data #####
    afq_feature=feature_generation(sss)    #### original data
#    afq_feature=combat_feature(sss)       #### harmonized data
    print(np.shape(afq_feature))
    
    feature=np.c_[popu[:,1:3],afq_feature]  #### add age and sex into the feature matrix
       
    ########### get ad&nc class #########
    mci_all=feature[np.where(label==2)[0],:]
    center_mci=center[np.where(label==2)]
    
    x=feature[np.where(label!=2)[0],:]  ### ADNC data
    y=label[np.where(label!=2)]
    center_adnc=center[np.where(label!=2)]
    label_adnc=y
    
    ########### training performance preparation ##########
    performance=dict()
    performance["acc"]=np.zeros((8,1))
    performance["sen"]=np.zeros((8,1))
    performance["spe"]=np.zeros((8,1))    
    performance["auc"]=np.zeros((8,1))
 
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_pred_all=np.empty(shape=(0,1))
    y_prob_all=np.empty(shape=(0,2))
    y_df_all=np.empty(shape=(0,1))
    
    mci_prob_all=np.empty(shape=(0,2))
    mci_df_all=np.empty(shape=(0,1))

    fea_wei=np.zeros((8,np.shape(x)[1]))  ### 8: 7 sites + 1 averaged
    
    for i in range(0,7):
        print("choose center %d as test set." %(i+1))
        
        ########## prepare train and test set ########
        x_test=x[np.where(center_adnc==i+1)[0],:]
        y_test=label_adnc[np.where(center_adnc==i+1)]
        
        x_train=x[np.where(center_adnc!=i+1)[0],:]
        y_train=label_adnc[np.where(center_adnc!=i+1)]
        
        mci=mci_all[np.where(center_mci==i+1)[0],:]

        ########## process missing values #############
        imputer= SimpleImputer(np.nan, "mean")
        imputer.fit(x_train)
        x_train=imputer.transform(x_train)
        x_test=imputer.transform(x_test)
        mci=imputer.transform(mci)    
        
        ################  scale  ##############
        scaler = StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test=scaler.transform(x_test)
        x_mci=scaler.transform(mci)
        
        ############ train & test ###########
        [y_pred,y_prob,y_df,fea_wei[i,:],model]=my_svm(x_train,y_train,x_test,y_test)
        [mci_prob,mci_df]=mci_test(x_mci,model)
        
        ############ evaluate ############
        [performance["acc"][i,0],performance["sen"][i,0],performance["spe"][i,0]]=my_model_performance(y_test,y_pred)
        fpr[i], tpr[i], _ = roc_curve(y_test,y_df,pos_label=3)
        performance["auc"][i,0]= auc(fpr[i], tpr[i])
        
        y_pred_all=np.append(y_pred_all,y_pred)
        y_prob_all=np.vstack((y_prob_all,y_prob))
        y_df_all=np.append(y_df_all,y_df)
        
        mci_prob_all=np.vstack((mci_prob_all,mci_prob))
        mci_df_all=np.append(mci_df_all,mci_df)

    performance["acc"][7,:]=np.mean(performance["acc"][0:7,:],axis=0) 
    performance["auc"][7,:]=np.mean(performance["auc"][0:7,:],axis=0)    
    performance["sen"][7,:]=np.mean(performance["sen"][0:7,:],axis=0) 
    performance["spe"][7,:]=np.mean(performance["spe"][0:7,:],axis=0) 
    fea_wei[7,:]=np.mean(fea_wei[0:7,:],axis=0) 
    
    fpr_avg,tpr_avg,_=roc_curve(label_adnc,y_df_all,pos_label=3)
    roc_auc_avg=auc(fpr_avg,tpr_avg)
    
#    with open('mix20_5_linearSVM.pkl', 'wb') as f:  
#        pickle.dump([performance, y_pred_all, y_prob_all, y_df_all, mci_prob_all, mci_df_all,fea_wei], f)    
   
    
    
    

    
        

