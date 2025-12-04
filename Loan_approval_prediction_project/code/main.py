import joblib
import os
import pandas as pd

class LoanApprovalApp:
    def __init__(self, classifier='stage_1_rf_classifier_pipeline.pkl', regressor="stage_2_rf_regression_pipeline.pkl"):

        # classifier : stage 1 model (0/1 prediction)
        # regressor  : stage 2 regression model

        model_path = os.path.join(os.getcwd(), "models")

        self.clf = joblib.load(os.path.join(model_path, classifier))
        self.reg = joblib.load(os.path.join(model_path, regressor))


        self.cols = [
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value"
        ]

    def get_user_input(self):
        print("--- Enter Applicant Details ---\n")
        # data = {}
        # data["no_of_dependents"] = int(input("No. of Dependents: "))
        # data["education"] = input("Education (Graduate/Not Graduate): ")
        # data["self_employed"] = input("Self Employed (Yes/No): ")
        # data["income_annum"] = float(input("Annual Income: "))
        # data["loan_amount"] = float(input("Loan Amount Requested: "))
        # data["loan_term"] = int(input("Loan Term (in years): "))
        # data["cibil_score"] = int(input("CIBIL Score: "))
        # data["residential_assets_value"] = float(input("Residential Assets Value: "))
        # data["commercial_assets_value"] = float(input("Commercial Assets Value: "))
        # data["luxury_assets_value"] = float(input("Luxury Assets Value: "))
        # data["bank_asset_value"] = float(input("Bank Asset Value: "))

        data = {
            "no_of_dependents": 3,
            "education": "Graduate",
            "self_employed": "Yes",
            "income_annum": 8300000,
            "loan_amount": 3001400000,
            "loan_term": 60,
            "cibil_score": 4,
            "residential_assets_value": 1000000,
            "commercial_assets_value": 1600000,
            "luxury_assets_value": 17200000,
            "bank_asset_value": 6100000
        }

        return pd.DataFrame([data])

    def two_stage_predict(self, applicant_df):

        # Stage 1 : Classifier --> Approve or Reject (0 or 1)
        # Stage 2 : Regression --> Predict optimal loan amount/interest (only if approved)

        out = {}

        applicant_df = applicant_df[self.clf.feature_names_in_]

        # print(applicant_df.columns.tolist())

        approve = self.clf.predict(applicant_df)[0]
        print(approve)
        out["approve"] = int(approve)

        if approve == 1:
            applicant_df_reg = applicant_df.copy()
            applicant_df_reg['loan_status'] = 'Approve'
            pred = self.reg.predict(applicant_df_reg)[0]
            out["regression_prediction"] = float(pred)
        else:
            out["regression_prediction"] = None

        return out

    def run(self):
        print("\n============================")
        print("  Loan Approval Application ")
        print("============================\n")

        # 1. Get user input
        applicant_df = self.get_user_input()

        # 2. Predict
        result = self.two_stage_predict(applicant_df)

        print("\n------ RESULT ------")
        if result["approve"] == 1:
            print("Loan Status : ✅ APPROVED")
            print(f"Predicted Loan Amount / Value : {result['regression_prediction']:.2f}")
        else:
            print("Loan Status : ❌ REJECTED")

        print("\n---------------------\n")
        return result
    
if __name__ == "__main__":

    loan_approval_application = LoanApprovalApp()
    loan_approval_application.run()