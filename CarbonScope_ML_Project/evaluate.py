import os, json, joblib, pandas as pd
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
BASE=os.path.dirname(os.path.abspath(__file__)); d=pd.read_csv(os.path.join(BASE,'dataset','carbon_emissions.csv')); NUM=['electricity_kwh','natural_gas_m3','vehicle_km','flights_km','waste_kg','renewable_pct','avg_temp_c','occupancy','industrial_output','transport_load']; CAT=['sector','region']; X=d[NUM+CAT]; y=d['carbon_kg_co2e']
def main():
 pipe=joblib.load(os.path.join(BASE,'models','gradient_boosting.joblib')); cv=KFold(5,shuffle=True,random_state=42); scores=cross_val_score(pipe,X,y,cv=cv,scoring='r2'); out={'gradient_boosting_cv_r2_mean':float(scores.mean()),'gradient_boosting_cv_r2_std':float(scores.std()),'grid_search':'learning_rate=[0.03,0.05], n_estimators=[100,150]'}; json.dump(out,open(os.path.join(BASE,'processed_data','evaluation_results.json'),'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
