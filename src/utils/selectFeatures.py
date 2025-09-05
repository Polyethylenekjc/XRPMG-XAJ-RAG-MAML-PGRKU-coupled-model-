import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def perform_granger_analysis(input_dir, output_dir, max_lag=3):
    """
    Performs Granger causality analysis on all CSV files in a directory to select
    the top 7 factors that have the most significant impact on 'runoffs'.
    
    Parameters:
    input_dir: Input directory for CSV files
    output_dir: Output directory for CSV files
    max_lag: Maximum lag for the Granger test
    """
    
    # Collect causality results from all files
    causality_results = defaultdict(list)
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Perform Granger causality analysis for each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)
        
        # Ensure the 'runoffs' column exists
        if 'runoffs' not in df.columns:
            print(f"The 'runoffs' column was not found in file {csv_file}, skipping.")
            continue
            
        # Get all feature columns except 'runoffs'
        feature_columns = [col for col in df.columns if col != 'runoffs']
        
        # Perform Granger causality test for each feature
        for feature in feature_columns:
            try:
                # Construct data for the Granger test (feature -> runoffs)
                test_data = df[[feature, 'runoffs']].dropna()
                
                if len(test_data) < max_lag + 1:
                    continue
                    
                # Execute the Granger causality test
                result = grangercausalitytests(test_data, max_lag, verbose=False)
                
                # Extract the minimum p-value as the causality strength of the feature on 'runoffs'
                # (smaller p-value indicates stronger causality)
                min_p_value = min([result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)])
                causality_results[feature].append(min_p_value)
                
            except Exception as e:
                print(f"An error occurred while processing {feature} in {csv_file}: {e}")
                continue
    
    # Calculate the average causality strength (p-value) for each feature
    avg_causality_strength = {}
    for feature, p_values in causality_results.items():
        avg_causality_strength[feature] = np.mean(p_values)
    
    # Sort by causality strength (smaller p-value is more important)
    sorted_features = sorted(avg_causality_strength.items(), key=lambda x: x[1])
    
    # Select the top 7 most influential features
    top_features = [feature for feature, _ in sorted_features[:7]]
    print(f"Selected top 7 features: {top_features}")
    
    # Process each CSV file and generate a new file containing only the selected features and 'runoffs'
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        
        df = pd.read_csv(input_path)
        
        # Select 'runoffs' and the selected feature columns
        selected_columns = ['runoffs'] + top_features
        available_columns = [col for col in selected_columns if col in df.columns]
        
        new_df = df[available_columns]
        new_df.to_csv(output_path, index=False)
        print(f"File generated: {output_path}")

# Example usage
perform_granger_analysis('/home/fifth/code/Python/GTLF/data/meta', '/home/fifth/code/Python/GTLF/data/select')
