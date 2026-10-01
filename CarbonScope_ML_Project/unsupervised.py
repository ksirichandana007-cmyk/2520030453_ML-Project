import os, json, joblib, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
BASE=os.path.dirname(os.path.abspath(__file__)); d=pd.read_csv(os.path.join(BASE,'dataset','carbon_emissions.csv')); NUM=['electricity_kwh','natural_gas_m3','vehicle_km','flights_km','waste_kg','renewable_pct','avg_temp_c','occupancy','industrial_output','transport_load']
def main():
 X=StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(d[NUM])); p=PCA(2,random_state=42).fit_transform(X); out={'kmeans_clusters':int(KMeans(4,random_state=42,n_init=10).fit(X).n_clusters),'agglomerative_clusters':int(AgglomerativeClustering(4).fit(X).n_clusters_),'dbscan_labels':int(len(set(DBSCAN(eps=1.3,min_samples=8).fit_predict(X)))),'pca_variance':PCA(2).fit(X).explained_variance_ratio_.tolist(),'anomalies':int((IsolationForest(contamination=.03,random_state=42).fit_predict(X)==-1).sum())}; json.dump(out,open(os.path.join(BASE,'processed_data','unsupervised_results.json'),'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
