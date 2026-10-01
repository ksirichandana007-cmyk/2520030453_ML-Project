import os, joblib, pandas as pd
BASE=os.path.dirname(os.path.abspath(__file__)); MODEL=os.path.join(BASE,'models','gradient_boosting.joblib')
def predict(values):
    return float(joblib.load(MODEL).predict(pd.DataFrame([values]))[0])
if __name__=='__main__':
    values={'electricity_kwh':500,'natural_gas_m3':100,'vehicle_km':800,'flights_km':1000,'waste_kg':50,'renewable_pct':30,'avg_temp_c':28,'occupancy':70,'industrial_output':100,'transport_load':80,'sector':'Commercial','region':'South'}
    print(f'Predicted emissions: {predict(values):,.2f} kg CO2e')
