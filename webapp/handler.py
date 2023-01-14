import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from health_insurance.HealthInsurance import HealthInsurance

# loading model
path = ''
model = pickle.load( open( path + 'model/lr_model.pkl', 'rb' ) )

# initialize API
app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] ) # Post é só para receber
def health_insurance_predict():
    # Recebe um json
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instantiate Rossmann class
        pipeline = HealthInsurance()
        
        # data cleaning
        df1 = pipeline.feature_engineering( test_raw )
        
        # feature engineering
        df2 = pipeline.pre_processing( df1 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df2 )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run( host='0.0.0.0', port=port )
    
