import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup style
try:
    sns.set_theme(style="whitegrid")
except ImportError:
    plt.style.use('ggplot')

# Colors matching the user's previous request order for the 4 folders
# We will assign these colors to the folders in alphabetical order
FOLDER_COLORS = ['#FFD700', 'red', 'blue', 'black'] # Gold, Red, Blue, Black

def parse_args():
    parser = argparse.ArgumentParser(description="Generate individual graphs for Train, Accuracy, and AP logs in each folder.")
    parser.add_argument('base_dir', help="Path to the base directory containing the 4 model folders.")
    parser.add_argument('--x_axis', choices=['steps', 'time'], default='steps', help="X-axis to use (default: steps)")
    return parser.parse_args()

def load_json_as_df(file_path):
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
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_single_graph(df, metric_name, folder_name, color, output_path, x_axis_choice):
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Data Mapping
    y_col = 'C'
    x_col = 'B' if x_axis_choice == 'steps' else 'A'
    
    # Plot
    plt.plot(df[x_col], df[y_col], color=color, linewidth=2.5)
    
    # Aesthetic Labels
    metric_title_map = {
        'train': 'Training Loss',
        'accuracy': 'Total Accuracy',
        'ap': 'Average Precision (AP)'
    }
    title = f"{folder_name} - {metric_title_map.get(metric_name, metric_name.capitalize())}"
    
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    
    x_label = "Steps" if x_axis_choice == 'steps' else "Time"
    plt.xlabel(x_label, fontsize=12)
    
    # Y Label
    y_label_map = {
        'train': 'Loss',
        'accuracy': 'Accuracy',
        'ap': 'AP'
    }
    plt.ylabel(y_label_map.get(metric_name, "Value"), fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, format='jpg', dpi=300)
    plt.close()
    print(f"Generated: {output_path}")

def main():
    args = parse_args()
    base_dir = args.base_dir
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    # metrics to look for
    metrics = ['train', 'accuracy', 'ap']
    
    # Get subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort() 
    
    print(f"Processing folders: {subdirs}")
    
    for i, folder in enumerate(subdirs):
        # User requested all graphs to be blue
        color = '#2874A6' # Nice standard plotting blue
        
        folder_path = os.path.join(base_dir, folder)
        
        for metric in metrics:
            filename = f"{metric}.json"
            
            # Find file (check root and jsonResultsData)
            file_path = os.path.join(folder_path, filename)
            if not os.path.exists(file_path):
                 nested_path = os.path.join(folder_path, "jsonResultsData", filename)
                 if os.path.exists(nested_path):
                     file_path = nested_path
            
            if os.path.exists(file_path):
                df = load_json_as_df(file_path)
                if df is not None and not df.empty:
                    # Output file inside the SAME folder
                    # Naming: graph_accuracy.jpg, graph_train.jpg, etc.
                    # Or just accuracy.jpg? "graph_" prefix is safer to avoid overwriting inputs if extensions differ,
                    # but here input is .json and output is .jpg so it's fine.
                    # User asked "each one in corresponding folder".
                    out_filename = f"graph_{metric}.jpg"
                    # If the json was in jsonResultsData, user might want the graph there or in the main model folder?
                    # "save each one in corresponding folder" usually means where the data is, or the model root.
                    # I'll save it in the MODEL ROOT (folder_path) so it's easy to see.
                    output_path = os.path.join(folder_path, out_filename)
                    
                    plot_single_graph(df, metric, folder, color, output_path, args.x_axis)
                else:
                    print(f"Skipping empty or invalid data: {file_path}")
            else:
                # Silent skip or warning?
                # User knows some might be missing.
                pass

if __name__ == "__main__":
    main()
