"""Paper 3 final robustness verification. Run with Python 3.10+ and installed requirements.
Input: five private model-ready CSV parts. Do not commit patient-level data.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

BASE=['log_review_count','log_patient_count','doctor_title_num','sentiment_continuous','complaint_present','gratitude_present']
REQ=['rating',*BASE,'informativeness','doctor_gender','department','review_year','doctor_id']

def run(folder, out, ordered=True):
    paths=[folder/f'paper3_model_ready_part{i}.csv' for i in range(1,6)]
    if not all(p.is_file() for p in paths): raise FileNotFoundError('Expected five input CSV files')
    df=pd.concat([pd.read_csv(p,low_memory=False) for p in paths],ignore_index=True)
    assert len(df)==30005 and df.review_id.is_unique
    df=df.dropna(subset=['review_id','doctor_id','rating']).copy()
    reg=df.dropna(subset=REQ).copy()
    assert len(reg)==28433, f'Unexpected complete-case N={len(reg)}'
    grouped=reg.department.value_counts()
    reg['department_group']=np.where(reg.department.isin(grouped[grouped>=500].index),reg.department,'Other')
    controls='+ C(doctor_gender) + C(department_group) + C(review_year)'
    rhs=' + '.join(BASE)+ ' '+controls
    results={'full_n':len(df),'complete_case_n':len(reg),'models':{}}
    reg['highest_rating']=(reg.rating==5).astype(int)
    for name,add in [('M4',''),('M5',' + informativeness')]:
        model=smf.logit('highest_rating ~ '+rhs+add,data=reg).fit(disp=False,maxiter=250,cov_type='cluster',cov_kwds={'groups':reg.doctor_id})
        results['models']['binary_'+name]={'n':int(model.nobs),'pseudo_r2':float(model.prsquared),'coefficients':{k:{'coef':float(model.params[k]),'se':float(model.bse[k]),'p':float(model.pvalues[k])} for k in BASE+(['informativeness'] if add else [])}}
        print('Binary',name,'pseudo-R2',model.prsquared,flush=True)
    formula='rating ~ '+' + '.join(BASE)+' + C(doctor_gender) + C(department) + C(review_year)'
    ols=smf.ols(formula,data=reg).fit(cov_type='cluster',cov_kwds={'groups':reg.doctor_id})
    resid=ols.resid
    jb=stats.jarque_bera(resid)
    results['residuals']={'n':len(resid),'r2':float(ols.rsquared),'skewness':float(stats.skew(resid,bias=False)),'kurtosis':float(stats.kurtosis(resid,fisher=False,bias=False)),'jarque_bera':float(jb.statistic),'jarque_bera_p':float(jb.pvalue)}
    print('Residuals',results['residuals'],flush=True)
    sens=df.copy()
    sens['doctor_title_missing']=sens.doctor_title_num.isna().astype(int)
    sens['doctor_title_num_filled']=sens.doctor_title_num.fillna(sens.doctor_title_num.median())
    sens['department_filled']=sens.department.fillna('Missing/Unknown').astype(str)
    sens['doctor_gender_filled']=sens.doctor_gender.fillna('Missing/Unknown').astype(str)
    srhs='log_review_count + log_patient_count + doctor_title_num_filled + doctor_title_missing + sentiment_continuous + complaint_present + gratitude_present + C(doctor_gender_filled) + C(department_filled) + C(review_year)'
    for name,add in [('M4',''),('M5',' + informativeness')]:
        model=smf.ols('rating ~ '+srhs+add,data=sens).fit(cov_type='cluster',cov_kwds={'groups':sens.doctor_id})
        results['models']['missing_title_'+name]={'n':int(model.nobs),'r2':float(model.rsquared),'adj_r2':float(model.rsquared_adj),'missing_title_coef':float(model.params['doctor_title_missing']),'missing_title_p':float(model.pvalues['doctor_title_missing'])}
        print('Missing-title',name,'N',model.nobs,'R2',model.rsquared,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'stage3_5_3_6_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    if ordered:
        dummies=pd.get_dummies(reg[['doctor_gender','department_group','review_year']].astype(str),drop_first=True,dtype=float)
        for name,add in [('M4',[]),('M5',['informativeness'])]:
            X=pd.concat([reg[BASE+add].astype(float),dummies],axis=1)
            model=OrderedModel(reg.rating.astype(int),X,distr='logit').fit(method='bfgs',disp=False,maxiter=1000)
            results['models']['ordered_'+name]={'n':len(reg),'converged':bool(model.mle_retvals.get('converged',False)),'coefficients':{k:{'coef':float(model.params[k]),'se':float(model.bse[k]),'p':float(model.pvalues[k])} for k in BASE+add}}
            (out/'stage3_5_3_6_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print('Ordered',name,'converged',model.mle_retvals.get('converged'),flush=True)
    print('Saved results:',out/'stage3_5_3_6_results.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('verification_results'))
    parser.add_argument('--skip-ordered',action='store_true',help='Skip slow ordered logistic models')
    a=parser.parse_args()
    run(a.data_dir,a.output_dir,ordered=not a.skip_ordered)
