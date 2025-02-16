csv_path = "Sample_data.csv"
with open(csv_path,'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i == 26:  # Since line numbers are 1-based
            print(f"Problematic line: {line}")
            break
