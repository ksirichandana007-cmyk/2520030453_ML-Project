import os, json, joblib, pandas as pd
from sklearn.inspection import permutation_importance
BASE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(BASE,'dataset','carbon_emissions.csv')
NUM=['electricity_kwh','natural_gas_m3','vehicle_km','flights_km','waste_kg','renewable_pct','avg_temp_c','occupancy','industrial_output','transport_load']; CAT=['sector','region']; TARGET='carbon_kg_co2e'
def main():
 d=pd.read_csv(DATA); print(d.describe(include='all').T); print('\nMissing:\n',d.isna().sum()); print('\nCorrelation:\n',d[NUM+[TARGET]].corr()[TARGET].sort_values(ascending=False));
 pipe=joblib.load(os.path.join(BASE,'models','gradient_boosting.joblib')); x=d[NUM+CAT].copy(); y=d[TARGET]; x=x.iloc[:300]; y=y.iloc[:300]; r=permutation_importance(pipe,x,y,n_repeats=5,random_state=42); imp=pd.DataFrame({'feature':NUM+CAT,'importance':r.importances_mean}).sort_values('importance',ascending=False); imp.to_csv(os.path.join(BASE,'processed_data','feature_importance.csv'),index=False); print('\nFeature importance:\n',imp)
if __name__=='__main__': main()
