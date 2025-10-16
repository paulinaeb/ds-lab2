import json

def clean_json_data():
    # Keys to keep (from the latest JSON object)
    keys_to_keep = {
        "Title",
        "Description", 
        "Primary Description",
        "Detail URL",
        "Location",
        "Skill",
        "Insight",
        "Job State",
        "Poster Id",
        "Company Name",
        "Company Logo",
        "Created At",
        "Scraped At"
    }
    
    try:
        # Read the original data
        print("Reading data.json...")
        with open('data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded {len(data)} records")
        
        # Clean each record
        cleaned_data = []
        for i, record in enumerate(data):
            if i % 1000 == 0:  # Progress indicator
                print(f"Processing record {i}...")
            
            # Keep only the specified keys
            cleaned_record = {key: record.get(key) for key in keys_to_keep if key in record}
            cleaned_data.append(cleaned_record)
        
        # Write the cleaned data
        print("Writing cleaned data to data_cleaned.json...")
        with open('data_cleaned.json', 'w', encoding='utf-8') as file:
            json.dump(cleaned_data, file, indent=2, ensure_ascii=False)
        
        print(f"Successfully cleaned {len(cleaned_data)} records")
        print("Cleaned data saved to 'data_cleaned.json'")
        
        # Show a sample of the first cleaned record
        if cleaned_data:
            print("\nSample of first cleaned record:")
            print(json.dumps(cleaned_data[0], indent=2))
            
    except FileNotFoundError:
        print("Error: data.json file not found!")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_json_data()