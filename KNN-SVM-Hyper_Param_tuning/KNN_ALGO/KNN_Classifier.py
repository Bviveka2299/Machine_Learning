import os
import pandas as pd
import joblib
import re

class CancerDiagnostics:
    
    def __init__(self, model="KNNClassifier.pkl"):
        
        model_path=os.path.join(os.getcwd(), "models")
        self.model=joblib.load(os.path.join(model_path, model))
        
        self.cols=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                   'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
                   'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
                   'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
                   'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                   'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                   'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
                   
    def get_user_input(self):
        print("--------------Enter details------------------")
        data={}
        
        def get_numeric_input(self, value):
            while True:
                try:
                    return float(value)
                except:
                    print("\nPlease enter valid input")
                    break
        
        for item in self.cols:
            result=get_numeric_input(self, input(f"{item}:"))
            if result:
                data[item]=result
            else:
                break
        return pd.DataFrame([data])
        
    def predict(self, input_df):
        
        out={}
        input_df=input_df[self.model.feature_names_in_]
        predict_diagnosis=self.model.predict(input_df)[0]
        print(predict_diagnosis)
        out['diagnosis']=predict_diagnosis
        
        df_for_predict=input_df.copy()
        
        if predict_diagnosis==1:
            
            df_for_predict['diagnosis']=1
            
        else:
            
            df_for_predict['diagnosis']=0
            
        return out
        
    def run(self):
        
        input_df=self.get_user_input()
        if input_df.shape[1]==30:
            result=self.predict(input_df)
            
            if result['diagnosis']==1:
                print("Status : ✅ Cancerous")
                
            else:
                print("Status : ❌ Non Cancerous")
                
            print("\n--------------------------\n")
        else:
            return
        return result
        
        
if __name__=="__main__":
    
    predict_tumour=CancerDiagnostics()
    predict_tumour.run()