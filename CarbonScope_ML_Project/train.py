import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(BASE,'dataset','carbon_emissions.csv')
MODELS=os.path.join(BASE,'models'); PROC=os.path.join(BASE,'processed_data')
os.makedirs(MODELS,exist_ok=True); os.makedirs(PROC,exist_ok=True)

features_num=['electricity_kwh','natural_gas_m3','vehicle_km','flights_km','waste_kg','renewable_pct','avg_temp_c','occupancy','industrial_output','transport_load']
features_cat=['sector','region']; target='carbon_kg_co2e'

def make_dataset(n=1500, seed=42):
    rng=np.random.default_rng(seed)
    sector=rng.choice(['Residential','Commercial','Industrial','Transport'],n,p=[.35,.25,.25,.15])
    region=rng.choice(['North','South','East','West','Central'],n)
    d=pd.DataFrame({
      'electricity_kwh':rng.gamma(5,140,n),'natural_gas_m3':rng.gamma(4,45,n),
      'vehicle_km':rng.gamma(5,180,n),'flights_km':rng.gamma(2.5,900,n),
      'waste_kg':rng.gamma(4,35,n),'renewable_pct':rng.uniform(0,100,n),
      'avg_temp_c':rng.normal(27,5,n),'occupancy':rng.uniform(10,100,n),
      'industrial_output':rng.gamma(5,70,n),'transport_load':rng.gamma(4,80,n),
      'sector':sector,'region':region})
    sf={'Residential':.85,'Commercial':1.0,'Industrial':1.35,'Transport':1.15}
    rf={'North':1.05,'South':.95,'East':1.02,'West':1.08,'Central':1.0}
    base=(.48*d.electricity_kwh+1.9*d.natural_gas_m3+.18*d.vehicle_km+.055*d.flights_km+
          1.15*d.waste_kg-.75*d.renewable_pct+.35*d.industrial_output+.25*d.transport_load+
          .35*d.occupancy+np.maximum(d.avg_temp_c-24,0)*3)
    d[target]=base*d.sector.map(sf)*d.region.map(rf)+rng.normal(0,90,n)
    d[target]=d[target].clip(lower=10).round(2)
    for col in ['natural_gas_m3','waste_kg','renewable_pct']:
        idx=rng.choice(n,size=int(.015*n),replace=False); d.loc[idx,col]=np.nan
    d.to_csv(DATA,index=False)
    return d

def main():
    d=make_dataset() if not os.path.exists(DATA) else pd.read_csv(DATA)
    X=d[features_num+features_cat]; y=d[target]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),features_num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),features_cat)])
    models={'Linear Regression':LinearRegression(),'Ridge':Ridge(alpha=1.0),'Lasso':Lasso(alpha=.1,max_iter=10000),'Elastic Net':ElasticNet(alpha=.1,l1_ratio=.5,max_iter=10000),'Decision Tree':DecisionTreeRegressor(max_depth=10,random_state=42),'Random Forest':RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(n_estimators=150,learning_rate=.05,max_depth=3,random_state=42)}
    rows=[]
    for name,m in models.items():
        pipe=Pipeline([('preprocess',pre),('model',m)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
        rows.append({'model':name,'mae':mean_absolute_error(yte,p),'rmse':mean_squared_error(yte,p)**.5,'r2':r2_score(yte,p)})
        joblib.dump(pipe,os.path.join(MODELS,name.lower().replace(' ','_')+'.joblib'))
    metrics=pd.DataFrame(rows).sort_values('r2',ascending=False); metrics.to_csv(os.path.join(PROC,'model_metrics.csv'),index=False)
    meta={'features_num':features_num,'features_cat':features_cat,'target':target,'train_rows':len(Xtr),'test_rows':len(Xte),'dataset_rows':len(d),'missing_values':int(d.isna().sum().sum())}
    with open(os.path.join(MODELS,'metadata.json'),'w') as f: json.dump(meta,f,indent=2)
    d.to_csv(os.path.join(PROC,'final_dataset.csv'),index=False)
    print(metrics.to_string(index=False)); print('Training complete.')
if __name__=='__main__': main()
