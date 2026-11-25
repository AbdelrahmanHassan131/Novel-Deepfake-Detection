import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Setup style
try:
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use('ggplot')

# User specific colors: Yellow, Red, Blue, Black
# Using Standard matplotlib named colors or hex codes
# 'gold' is often better than 'yellow' for reading on white, but we will stick close to request.
CUSTOM_COLORS = ['#FFD700', 'red', 'blue', 'black']  # Gold/Yellow, Red, Blue, Black

def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison graphs for Train, Accuracy, and AP logs.")
    parser.add_argument('base_dir', help="Path to the base directory containing the 4 model folders.")
    parser.add_argument('--output_dir', default=None, help="Directory to save output images. Defaults to base_dir.")
    parser.add_argument('--x_axis', choices=['steps', 'time'], default='steps', help="X-axis to use (default: steps)")
    return parser.parse_args()

def load_json_as_df(file_path):
    """
    Loads JSON and converts to DataFrame with standardized columns [A, B, C].
    A: Time, B: Steps, C: Value
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            return None

        # Handle [Time, Steps, Value] list of lists
        if list(df.columns) == [0, 1, 2]:
            df.columns = ['A', 'B', 'C']
        
        # Ensure we have A, B, C keys now (if it was dict with keys "A","B","C", it deals with itself)
        # If keys are different, we might fail, but assuming consistent format with previous task.
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_metric_comparison(metric_name, folder_data_map, output_path, x_axis_choice='steps'):
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Cycle through our custom colors
    color_cycle = iter(CUSTOM_COLORS)
    
    plotted_any = False
    
    for folder_name, df in folder_data_map.items():
        if df is None or df.empty:
            continue
            
        # Determine X and Y
        # A=Time, B=Steps, C=Value
        y_col = 'C'
        x_col = 'B' if x_axis_choice == 'steps' else 'A'
        
        # Get color (loop if more than 4, currently just 4 expected)
        try:
            color = next(color_cycle)
        except StopIteration:
            color_cycle = iter(CUSTOM_COLORS)
            color = next(color_cycle)
            
        # Rename columns for semantic legend
        # We plot directly
        
        label = f"{folder_name}"
        
        # Smoothing could be nice, but raw data is requested.
        plt.plot(df[x_col], df[y_col], label=label, color=color, linewidth=2)
        plotted_any = True

    if not plotted_any:
        print(f"No valid data found for {metric_name}. Skipping plot.")
        plt.close()
        return

    plt.title(f"Comparison: {metric_name.capitalize()}", fontsize=18, fontweight='bold', pad=20)
    plt.ylabel("Value", fontsize=14)
    plt.xlabel("Steps" if x_axis_choice == 'steps' else "Time", fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved {metric_name} graph to: {output_path}")
    plt.close()

def main():
    args = parse_args()
    
    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else base_dir
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    # metrics to look for
    metrics = ['train', 'accuracy', 'ap']
    
    # Get subdirectories
    # We filter for only directories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort() # Ensure consistent order for colors if needed
    
    print(f"Found folders: {subdirs}")
    
    # Loop over each metric and generate a graph
    for metric in metrics:
        filename = f"{metric}.json"
        
        # Collect data for this metric from all folders
        folder_data_map = {}
        
        for folder in subdirs:
            file_path = os.path.join(base_dir, folder, filename)
            
            # Check if file exists (it might be in a nested 'jsonResultsData' folder like previous example? 
            # User said "contain 3 files json accuracy and ap and train", implying direct containment.
            # However, previous path was G:\checkpoints\Wang202_128\jsonResultsData\accuracy.json
            # Use a smart check: check direct, then check jsonResultsData subdir.
            
            if not os.path.exists(file_path):
                # Try common subdir pattern
                 nested_path = os.path.join(base_dir, folder, "jsonResultsData", filename)
                 if os.path.exists(nested_path):
                     file_path = nested_path
            
            if os.path.exists(file_path):
                df = load_json_as_df(file_path)
                if df is not None:
                    folder_data_map[folder] = df
            else:
                print(f"Warning: {filename} not found in {folder}")

        # Plot
        if folder_data_map:
            out_file = os.path.join(output_dir, f"compare_{metric}.jpg")
            plot_metric_comparison(metric, folder_data_map, out_file, args.x_axis)
        else:
            print(f"No data found for metric: {metric}")

if __name__ == "__main__":
    main()
