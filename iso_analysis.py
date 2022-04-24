import numpy as np
import pandas as pd
import regex as re
from collections import Counter
import random
random.seed(10)
import time
begin=time.time()

##-----------------------------------------------
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, Y = data.data, data.target  
bc_fn= [a.replace(' ', '_') for a in data.feature_names]
print('\nDataset: %s'%str(X.shape))
print('Targets: 0(malignant-%d recs); 1(benign-%d recs)\n'%(sum(Y==0),sum(Y==1)))
# TARGETS: 
#    Malignant(outlier):0   -- 212 records (37%)
#    Benign:1               -- 357 records
##-----------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
leaf_val = '|---value:' # id at every leaf when parsing export_text
nmodels = 20
ntrees = 250
outlr_fract = 0.37      # contamination
pth_len = 3             # thld for length for short paths
print('Models %d | Trees %d | Outlier Fraction %.2f | Short Path length %d\n'
      %(nmodels,ntrees,outlr_fract,pth_len))
rng_lst = random.sample(range(1,200), nmodels) # rand state for each forest
mnames=['M'+str(i+1) for i in range(0,nmodels)]
##-----------------------------------------------
#  extract all paths from a tree
def recurse_tree(tlist, flst, plst, pos):
    if leaf_val in tlist[pos]:
        flst.append(plst.copy())
        if pos == len(tlist) - 1: return flst
        while plst[-1].split(' ')[0] != tlist[pos+1].split(' ')[0]: 
            plst.pop()
        plst.pop()
        pos = pos + 1
    else:
        plst.append(tlist[pos])
        pos = pos + 1
    flst = recurse_tree(tlist, flst, plst, pos)
    return flst
##-----------------------------------------------
# for a range of random states, generate Isolation forests,
# extract short paths from each forest, compute feature frequencies
exp_results=[]
for rng in rng_lst:
    clf = IsolationForest(n_estimators=ntrees#, max_samples=len(X)
                          , random_state=rng, contamination=outlr_fract
                          , bootstrap=True)
    clf.fit(X)
    y_pred = clf.predict(X)
    y_pred = np.where(y_pred==-1,0,1)
    out_count = y_pred.tolist().count(0)
    tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()
    prcsn = tp/(tp+fp)
    rcll = tp/(tp+fn)
    f1 = ((2*prcsn*rcll)/(prcsn+rcll))
    print("Model %d | Precision %.2f | Recall %.2f | F1 %.2f"
          %(rng,prcsn,rcll,f1))
    ##-----------------------------------------------
    # for each model/forest extract short paths from all trees
    flist = []
    for t in range(ntrees):
        dt = export_text(clf.estimators_[t], feature_names=bc_fn)
        dt_list = dt.replace('|   ','|').replace('|--- ', '|---').split('\n')[:-1]
        dt_list_leaves = sum([1 for d in dt_list if re.findall(r'\\'+leaf_val, d)])
        flst = recurse_tree(dt_list, [], [], 0)
        flst_len=len(flst)
        flst=[fl for fl in flst if len(fl)<=pth_len]  # prune for short paths only
        # print('Tree %d / Leaves %d / Paths %d / Short Paths %d'%(t,dt_list_leaves,flst_len,len(flst)))
        if(dt_list_leaves != flst_len): 
            print('   *** Extraction mismatch (uncomment above print stmt)')
        flist.extend(flst)

    print('Short paths extracted %d'%len(flist))
    ##-----------------------------------------------
    # for each model/forest extract top 5 outliers
    sscores = list(np.argsort(clf.score_samples(X))[:5])
    exp_results.append(sscores)
   
    # for the top outlier in each model, extract its top 5 features
    tp_outlr=sscores[0]
    dic_outlr_val={}
    for col,val in zip(bc_fn,X[tp_outlr]): dic_outlr_val[col]=val
    flist=[[re.sub(r'\|+---','',a) for a in fl] for fl in flist]
    flist_eval=[]
    for fl in flist:
        flt=[]
        for f in fl:
            ft=re.sub(r' +',' ',f)
            col,opnd,num=ft.split(' ')
            cval=str(dic_outlr_val[col])
            ft=''.join(['(',cval,opnd,num,')'])
            flt.append(ft)
        flist_eval.append(flt)
    flist_eval_res=[eval(' & '.join(fl)) for fl in flist_eval]
    print('Top Outlier idx %d (Target %d) Short paths %d\n'
                  %(tp_outlr,Y[tp_outlr],sum(flist_eval_res)))
    tp_outlr_flist=[f for i,f in enumerate(flist) if flist_eval_res[i]]
    tp_outlr_ffreq=[f.split(' ')[0] for fl in tp_outlr_flist for f in fl]
    tp_outlr_ffreq=[x[0] for x in Counter(tp_outlr_ffreq).most_common(5)]
    exp_results.append(tp_outlr_ffreq)

##-----------------------------------------------
# aggregate results into rankings for each model's top outliers
# and the top outlier's top features
# print(exp_results)
out_lst=[x for i,x in enumerate(exp_results) if (i%2)==0]
out_lst_unpk=set([x for rl in out_lst for x in rl])
out_df=pd.DataFrame(np.zeros((len(out_lst_unpk),len(mnames)+1))\
                    , index=out_lst_unpk, columns=mnames+['rank'], dtype=np.int32)
for i in out_lst_unpk:
    for j in mnames:
        rng_out_lst=out_lst[mnames.index(j)]
        out_df.loc[i,j]=rng_out_lst.index(i)+1 if i in rng_out_lst else 99
out_df['rank']=out_df.sum(axis=1)
out_df=out_df.sort_values(by='rank')
print('Outlier ranks:')
print(out_df)
out_df.to_csv("outlier_ranks_"+str(ntrees)+".csv")

tp_outlr_feat_lst=[x for i,x in enumerate(exp_results) if (i%2)==1]
tp_outlr_feat_unpk=set([x for fl in tp_outlr_feat_lst for x in fl])
tp_outlr_feat_df=pd.DataFrame(np.zeros((len(tp_outlr_feat_unpk),len(mnames)+1))\
                              , index=tp_outlr_feat_unpk, columns=mnames+['rank'], dtype=np.int32)
for i in tp_outlr_feat_unpk:
    for j in mnames:
        rng_feat_lst=tp_outlr_feat_lst[mnames.index(j)]
        tp_outlr_feat_df.loc[i,j]=rng_feat_lst.index(i)+1 if i in rng_feat_lst else 99
tp_outlr_feat_df['rank']=tp_outlr_feat_df.sum(axis=1)
tp_outlr_feat_df=tp_outlr_feat_df.sort_values(by='rank')
print('\nFeature ranks for Top Outlier in each model:')
print(tp_outlr_feat_df)
tp_outlr_feat_df.to_csv("tp_outlr_feature_ranks_"+str(ntrees)+".csv")

print('\nTime taken: %.2f'%(time.time()-begin))
