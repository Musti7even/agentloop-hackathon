import json
import csv
import os

def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert JSON results file to CSV format, excluding metadata.
    
    Args:
        json_file_path (str): Path to the JSON file
        csv_file_path (str): Path to save the CSV file
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract results (excluding metadata)
    results = data.get('results', [])
    
    # Prepare CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV headers
        fieldnames = ['persona', 'message', 'decision', 'reasoning', 'expected_decision']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for result in results:
            persona = result.get('persona', '')
            messages = result.get('messages', [])
            
            for message_data in messages:
                message = message_data.get('message', '')
                decision_data = message_data.get('decision', {})
                decision = decision_data.get('decision', '')
                reasoning = decision_data.get('reasoning', '')
                
                
                # Write row to CSV
                writer.writerow({
                    'persona': persona,
                    'message': message,
                    'decision': str(decision),
                    'reasoning': reasoning,
                    'expected_decision': True
                })

if __name__ == "__main__":
    # File paths
    json_file = "data/results/b2b_saas_personas_train_results_20250802_144126.json"
    csv_file = "data/results/b2b_saas_personas_train_results_20250802_144126.csv"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Convert JSON to CSV
    convert_json_to_csv(json_file, csv_file)
    print(f"Conversion complete. CSV file saved to: {csv_file}")
