import pandas as pd

def create_exploration_csv():
    # Read the original clinical dataset
    df = pd.read_csv('clinical_dataset.csv')
    
    # The requirement is 100 samples for the exploration dataset
    # We take the first 100 samples as typically done in the /dataset endpoint in app.py
    df_subset = df.head(100)
    
    # Save to the new CSV file
    output_filename = 'Clinical Dataset Exploration.csv'
    df_subset.to_csv(output_filename, index=False)
    
    print(f"Successfully created '{output_filename}' with {len(df_subset)} samples.")

if __name__ == "__main__":
    create_exploration_csv()
