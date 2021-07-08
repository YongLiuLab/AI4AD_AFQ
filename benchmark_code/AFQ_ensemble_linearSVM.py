# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 20:21:59 2021

@author: dell
"""

import numpy as np
from sklearn import svm  
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from scipy import stats
import pickle



def my_svm(x_train,y_train,x_test,y_test):
    model = svm.SVC(kernel='linear',probability=True)  # linear-kernal, probability=True
    
    # GridSearchCV for optimal parameters, 3-fold CV
#    c_range = np.logspace(-15,-5, 15, base=2)
    c_range=np.array([0.00007,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
                      0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07])
    param_grid = [{'kernel': ['linear'], 'C': c_range}]
    gridsearch = GridSearchCV(model, param_grid, cv=3, n_jobs=4)
    
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
    print("------------------------processing features------------------------")
    for j in range(0,len(ll)):
        print('processing measure:',ll[j])
        for i in range(0,20):
            print('fiber %s'%(i+1))
            a=alldata[i][0][ll[j]]
            for k in range(0,int(100/smooth)):
                temp1=a[:,k*smooth:(k+1)*smooth]
                temp2=np.nanmean(temp1,1)
                out=np.hstack((out,temp2.reshape(825,1)))
    
    print("-------------------------------------------------------------------")
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



def my_LR(x_train,y_train,x_test,y_test,CR):
    #### 3-fold CV for optinal parameters #####
    C_range=CR
    lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=C_range,cv=3,penalty="l2",solver="lbfgs",tol=0.0001)
    ## training 
    re = lr.fit(x_train,y_train)
    
    y_pred = re.predict(x_test)
    score_train = re.score(x_train, y_train)
    score_test = re.score(x_test, y_test)
    y_prob=re.predict_proba(x_test)   
    y_df=re.decision_function(x_test)
    
    print("Logistic Regression:")
    print("trainset acc is: %s, testset acc is: %s"%(score_train,score_test))
    weight=re.coef_
    print("C is: %s, intercept is: %s, feature sparse ratio:%.2f%%" %(re.C_,re.intercept_,np.mean(lr.coef_.ravel()==0)*100))
    
    return y_pred,y_prob,y_df,weight,re




    


if __name__ == '__main__' :
    file = 'demography.mat'
    data = loadmat(file, mat_dtype=True)
    center=data['center']
    label=data['label']
    popu=data['popu']
    
    ll=['FA','MD','RD','AD','CL','VOLUME']
    
    ##### set average-smooth kernel ####
    sss=5  
    
    ##### choose original data or harmonized data #####
    afq_feature=feature_generation(sss)  #### original data
#    afq_feature=combat_feature(sss)       #### harmonized data
    print("total feature dim:")
    print(np.shape(afq_feature))

    size=20*100/sss     ##### the dimension of features
    
    ##### initialization #####
    y_pred_allms=np.empty(shape=(570,0))
    y_prob_allms=np.empty(shape=(570,0))
    y_df_allms=np.empty(shape=(570,0))
    
    mci_prob_allms=np.empty(shape=(255,0))
    mci_df_allms=np.empty(shape=(255,0))    
    
    performance=dict()
    performance["acc"]=np.zeros((8,8))  #### 8(7 sites + 1 avg) * 8(6 measures + hardvote + LR)
    performance["sen"]=np.zeros((8,8))
    performance["spe"]=np.zeros((8,8))    
    performance["auc"]=np.zeros((8,8))

    fi_fiber_all=dict()                 #### fiber importance
    fi_fiber_all["FA"]=np.zeros((int(size+2),8))   #### features+age&sex * 8(7 sites + 1 avg)
    fi_fiber_all["MD"]=np.zeros((int(size+2),8))
    fi_fiber_all["RD"]=np.zeros((int(size+2),8))    
    fi_fiber_all["AD"]=np.zeros((int(size+2),8))
    fi_fiber_all["CL"]=np.zeros((int(size+2),8))
    fi_fiber_all["VOLUME"]=np.zeros((int(size+2),8))
    
    
    ####### labels #######
    y=label[np.where(label!=2)]
    center_adnc=center[np.where(label!=2)]
    label_adnc=y
    center_mci=center[np.where(label==2)]
    
    ############### train for every measures and every center ##############
    print("-------------------- train for each measures ----------------------")
    for kk in range(0,len(ll)):
        ########### select a measure ############
        print("")
        print('############# training using:',ll[kk])
        print("%s and %s"%(int(kk*size),int((kk+1)*size)))
        measure=afq_feature[:,int(kk*size):int((kk+1)*size)]
        
        feature=np.c_[popu[:,1:3],measure]   ### add age and sex into feature matrix
        print("single measure feature dim:")
        print(np.shape(feature))
    
        ########### get ad&nc class #########
        mci_all=feature[np.where(label==2)[0],:]
        x=feature[np.where(label!=2)[0],:]
   
        ########### training performance preparation ##########          
        fpr = dict()
        tpr = dict()
        y_pred_all=np.empty(shape=(0,1))
        y_prob_all=np.empty(shape=(0,2))
        y_df_all=np.empty(shape=(0,1))
        
        mci_prob_all=np.empty(shape=(0,2))
        mci_df_all=np.empty(shape=(0,1))
    
        ############ train for every center ############
        for i in range(0,7):
            print("### choose center %d as test set." %(i+1))
            
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
                       
            ############ training & testing ###########
            [y_pred,y_prob,y_df,fi,model]=my_svm(x_train,y_train,x_test,y_test)
            [mci_prob,mci_df]=mci_test(x_mci,model)
            
            ############ evaluate ############
            [performance["acc"][i,kk],performance["sen"][i,kk],performance["spe"][i,kk]]=my_model_performance(y_test,y_pred)
            fi_fiber_all[ll[kk]][:,i]=fi            
            fpr[i], tpr[i], _ = roc_curve(y_test,y_df,pos_label=3)
            performance["auc"][i,kk]= auc(fpr[i], tpr[i])
            
            y_pred_all=np.append(y_pred_all,y_pred)
            y_prob_all=np.vstack((y_prob_all,y_prob))
            y_df_all=np.append(y_df_all,y_df)
            
            mci_prob_all=np.vstack((mci_prob_all,mci_prob))
            mci_df_all=np.append(mci_df_all,mci_df)
        
            del y_pred,y_prob,y_df,mci_prob,mci_df,x_train,x_test
            
        ########## make preds and probs from different measures together ##########
        y_pred_allms=np.hstack((y_pred_allms,y_pred_all.reshape(-1,1)))
        y_prob_allms=np.hstack((y_prob_allms,y_prob_all))
        y_df_allms=np.hstack((y_df_allms,y_df_all.reshape(-1,1)))
        
        mci_prob_allms=np.hstack((mci_prob_allms,mci_prob_all))
        mci_df_allms=np.hstack((mci_df_allms,mci_df_all.reshape(-1,1)))            
        
        fi_fiber_all[ll[kk]][:,7]=np.mean(fi_fiber_all[ll[kk]][:,0:7],axis=1)
        del y_pred_all,y_prob_all,y_df_all,mci_prob_all,mci_df_all
    
    print("--------------------------------------------------------------")
    print("")
    print("")    
    
    ###################### ensemble learning ########################
    print("-------------------- ensemble learning -----------------------")
    ########## hard vote #########
    vote=stats.mode(y_pred_allms,axis=1)[0]

    mea_weight=np.zeros((8,6))    ### 8:7 sites+1 average,  6:6 measures*1
    y_pred_LR_all=np.empty(shape=(0,1))
    y_prob_LR_all=np.empty(shape=(0,2))
    y_df_LR_all=np.empty(shape=(0,1))
    
    mci_prob_LR_all=np.empty(shape=(0,2))
    mci_df_LR_all=np.empty(shape=(0,1))    
    
    C_range_LR=np.array([0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5])
    C_range_LR=np.append(C_range_LR, np.logspace(0,4,15,base=2))
    ########## for each site #########
    for i in range(0,7):
        print("")
        print("### choose center %d as test set." %(i+1))
            
        test_LR=y_df_allms[np.where(center_adnc==i+1)[0],:]
        
        train_LR=y_df_allms[np.where(center_adnc!=i+1)[0],:]
        
        y_test=label_adnc[np.where(center_adnc==i+1)]
        y_train=label_adnc[np.where(center_adnc!=i+1)]
        
        x_mci=mci_df_allms[np.where(center_mci==i+1)[0],:]
        
        ################  scale  ##############
        scaler = StandardScaler().fit(train_LR)
        train_LR=scaler.transform(train_LR)
        test_LR=scaler.transform(test_LR)
        x_mci=scaler.transform(x_mci)
     
        ######## hard vote evaluate #######
        y_pred_vote=vote[np.where(center_adnc==i+1)]
        [performance["acc"][i,6],performance["sen"][i,6],performance["spe"][i,6]]=my_model_performance(y_test,y_pred_vote)
        
        ############# LR training ############        
        [y_pred_LR,y_prob_LR,y_df_LR, wei,model]=my_LR(train_LR,y_train,test_LR,y_test,C_range_LR)
        mea_weight[i,:]=wei
        [performance["acc"][i,7],performance["sen"][i,7],performance["spe"][i,7]]=my_model_performance(y_test,y_pred_LR)
        
        y_pred_LR_all=np.append(y_pred_LR_all,y_pred_LR)
        y_prob_LR_all=np.vstack((y_prob_LR_all,y_prob_LR))
        y_df_LR_all=np.append(y_df_LR_all,y_df_LR)
        
        
        fpr, tpr, _ = roc_curve(y_test,y_prob_LR[:,1],pos_label=3)
        performance["auc"][i,7]= auc(fpr, tpr)
        
        [mci_prob_LR,mci_df_LR]=mci_test(x_mci,model)
        mci_prob_LR_all=np.vstack((mci_prob_LR_all,mci_prob_LR))
        mci_df_LR_all=np.append(mci_df_LR_all,mci_df_LR)
             
        
    performance["acc"][7,:]=np.mean(performance["acc"][0:7,:],axis=0) 
    performance["auc"][7,:]=np.mean(performance["auc"][0:7,:],axis=0)    
    performance["sen"][7,:]=np.mean(performance["sen"][0:7,:],axis=0) 
    performance["spe"][7,:]=np.mean(performance["spe"][0:7,:],axis=0)    
    mea_weight[7,:]=np.mean(mea_weight[0:7,:],axis=0)          
    
#    print("saving")
#    with open('out.pkl', 'wb') as f:  
#        pickle.dump([performance, y_pred_allms, y_prob_allms, y_df_allms, mci_prob_allms, mci_df_allms, y_pred_LR_all, y_prob_LR_all, y_df_LR_all,
#                     mci_prob_LR_all, mci_df_LR_all, fi_fiber_all, mea_weight], f)
#    print("Done")
###    # Getting back the objects:
#    with open('out.pkl','rb') as f:  
#       [performance, y_pred_allms, y_prob_allms, y_df_allms, mci_prob_allms, mci_df_allms, y_pred_LR_all, y_prob_LR_all, y_df_LR_all,
#                     mci_prob_LR_all, mci_df_LR_all, fi_fiber_all, mea_weight]= pickle.load(f)  
    
