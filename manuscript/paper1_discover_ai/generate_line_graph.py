import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

# Aesthetic setup
try:
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use('ggplot')

def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON data to a Line Graph (JPG)")
    parser.add_argument('json_file', help="Path to the input JSON file")
    parser.add_argument('--x_axis', choices=['time', 'steps'], default='time', help="Choose X-axis: 'time' (Column A) or 'steps' (Column B)")
    parser.add_argument('--title', type=str, default="Line Graph", help="Title of the chart")
    parser.add_argument('--output', type=str, default="output_graph.jpg", help="Path to save the output JPG image")
    return parser.parse_args()

def load_data(json_file):
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)
    
    try:
        # Try reading as valid JSON first
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Normalize to DataFrame
        # Case 1: List of dicts [{"A":.., "B":.., "C":..}, ...]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        # Case 2: Dict of lists {"A": [..], "B": [..], "C": [..]}
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported JSON format. Expected list of lists, list of dicts, or dict of lists.")
            
        # Handle list of lists where columns are numeric indices 0, 1, 2
        # We expect 0->A, 1->B, 2->C based on user description
        if list(df.columns) == [0, 1, 2]:
            df.columns = ['A', 'B', 'C']
            print("Detected list-of-lists format. Mapped columns: 0->A, 1->B, 2->C")

        # Standardize column names if they map to intended A, B, C but have different names?
        # User specified "A", "B", "C" specifically, so we assume these keys exist now.
        required_cols = ['A', 'B', 'C']
        if not all(col in df.columns for col in required_cols):
             print(f"Warning: Expected columns {required_cols}, found {df.columns}. Chart generation might fail.")
        
        return df

    except Exception as e:
        print(f"Error reading JSON: {e}")
        sys.exit(1)

def plot_graph(df, x_choice, title, output_file):
    plt.figure(figsize=(12, 7), dpi=300)
    
    # Map selection to column name
    x_col = 'A' if x_choice == 'time' else 'B'
    y_col = 'C'

    # Pre-processing X-axis if it is Time (A)
    if x_choice == 'time':
        # Attempt to convert to datetime
        try:
            # Check if it looks like a unix timestamp (large floats)
            if pd.api.types.is_numeric_dtype(df[x_col]) and df[x_col].mean() > 1e9:
                 df[x_col] = pd.to_datetime(df[x_col], unit='s')
            else:
                 df[x_col] = pd.to_datetime(df[x_col])
            is_datetime = True
        except Exception as e:
            print(f"Time conversion warning: {e}. Treating as numeric/string.")
            # If not convertable (e.g. simple numeric ticks), treat as numeric
            is_datetime = False
    else:
        is_datetime = False

    # Plotting
    # Using seaborn for a nicer default look if available, else standard plt
    try:
        sns.lineplot(data=df, x=x_col, y=y_col, linewidth=2.5, color='#2874A6') # nice blue
    except NameError:
        plt.plot(df[x_col], df[y_col], linewidth=2.5, color='#2874A6')

    # Formatting
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    x_label = "Time" if x_choice == 'time' else "Steps"
    plt.xlabel(x_label, fontsize=12, labelpad=10)
    plt.ylabel("Value", fontsize=12, labelpad=10)

    # Ticks formatting for Time
    if x_choice == 'time' and is_datetime:
        ax = plt.gca()
        # User requested minutes and seconds. Using %H:%M:%S to ensure context is preserved,
        # but emphasizing the time component.
        formatter = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate() # Rotation
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
    print(f"Graph saved successfully to: {output_file}")
    plt.close()

def main():
    args = parse_args()
    df = load_data(args.json_file)
    plot_graph(df, args.x_axis, args.title, args.output)

if __name__ == "__main__":
    main()
