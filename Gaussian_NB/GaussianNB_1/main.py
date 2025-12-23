import os
os.system("cls")
import joblib
import re
import pandas as pd 

class IncomePrediction:
    
    def __init__(self, model="GaussianNB.pkl"):
        
        model_path=os.path.join(os.getcwd(), "models")
        self.model=joblib.load(os.path.join(model_path, model))
        
        self.cols=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
        self.num_cols=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        self.cat_cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
        
    def get_user_input(self):
        print("--------------Enter details------------------")
        data={}
        
        def get_numeric_input(self, value):
            try:
                return int(value)
                    
            except:
                print("\nPlease enter valid input")
                return 
                
        #print(len(self.num_cols))
        for col in self.num_cols:
            input_val=input(f"Please enter the value for {col}:")
            if input_val==0 or input_val:
                result=get_numeric_input(self, input_val)
                if result==0 or result:
                    data[col]=result
                else:
                    break
            else:
                print("\nInput_val should not be empty")
                break 
        if len(data)!=len(self.num_cols):
            return {'Novals':'Dummy'}
        for col in self.cat_cols:
            value=input(f"Please enter the input value for {col}:")
            if value:
                if re.fullmatch(r"[0-9A-Za-z \-]+", value):
                    data[col]=value
                else:
                    print("Please enter valid input")
                    break
            else:
                print("Value should not be empty")
                break
        if len(data)==len(self.cols):
            ordered_data={k:data[k] for k in self.cols}
        else:
            return {'Novals':'Dummy'}
        return ordered_data
        
    def predict(self, input_df):
        
        out={}
        input_df=input_df[self.model.feature_names_in_]
        predict_income=self.model.predict(input_df)[0]
        print(predict_income)
        out['income']=predict_income
        df_for_predict=input_df.copy()
        
        if predict_income==1:
            df_for_predict['income']='>50K'
        else:
            df_for_predict['income']='<=50K'
            
        return out
        
    def run(self):
        
        input_df=self.get_user_input()
        
        if len(input_df)==len(self.cols):
            result=self.predict(pd.DataFrame([input_df]))
            if result['income']==1:
                print("Income is >50K")
            else:
                print("Income is <=50K")
                
            print("\n--------------------\n")
        else:
            return
            
        return result
        
        
if __name__=="__main__":
    
    predict_income=IncomePrediction()
    predict_income.run()
            