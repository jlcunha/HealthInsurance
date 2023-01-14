import pickle
import numpy as np
import pandas as pd

class HealthInsurance( object ):
    
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaller = pickle.load( open( self.home_path + 'parameter/annual_premium.pkl', 'rb') )
        self.age_scaller =            pickle.load( open( self.home_path + 'parameter/age.pkl', 'rb') )
        self.vintage_scaller =        pickle.load( open( self.home_path + 'parameter/vintage.pkl', 'rb') )
        self.gender_scaller =         pickle.load( open( self.home_path + 'parameter/gender.pkl', 'rb') )
        self.region_code_scaller =    pickle.load( open( self.home_path + 'parameter/region_code.pkl', 'rb') )
        self.policy_sales_scaller =   pickle.load( open( self.home_path + 'parameter/policy_sales_channel.pkl', 'rb') )
        
#########################################################################################################################
    def feature_engineering( self, data):
        # vehicle_age
        data['vehicle_age'] = data['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year'
                                                                        if x == '1-2 Year' else 'below_1_year' )
        # vehicle_damage
        data['vehicle_damage'] = data['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0 )
        
        return( data )

#########################################################################################################################
    def pre_processing( self, data):
        # Standardization
        # Annual Premium
        data['annual_premium'] = self.annual_premium_scaller.transform( data[['annual_premium']].values )

        # Rescaling
        # Age
        data['age'] = self.age_scaller.transform( data[['age']].values )
        
        # Vintage
        data['vintage'] = self.vintage_scaller.transform( data[['vintage']].values )        

        # Encoder
        # Gender
        data.loc[:, 'gender'] = data['gender'].map( self.gender_scaller )

        # Region Code 
        data.loc[:, 'region_code'] = data['region_code'].map( self.region_code_scaller )
        
        # Vehicle Age
        data = pd.get_dummies( data, prefix='vehicle_age', columns=['vehicle_age'] )

        # Policy Sales Channel
        data.loc[:, 'policy_sales_channel'] = data['policy_sales_channel'].map( self.policy_sales_scaller )

        # Feature selection
        cols_selected = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage', 'previously_insured', 'policy_sales_channel']
        
        return( data[ cols_selected ])

#########################################################################################################################
    def get_prediction( self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json( orient='records', date_format='iso' )
