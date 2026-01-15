import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd

# Add project root to path to import castle modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from castle.ui.plot_mask_info import Plotter

# Base directory for the project
BASE_DIR = '/mnt/AB-VFM/castle-project/2024-09-11-01-37-06-B3D71S21S40-session-level'
TRACK_DIR = os.path.join(BASE_DIR, 'track')

def get_roi_data(h5_path):
    """
    Reads mask_list.h5 and returns data structured for Plotter.
    Assumes single ROI for now (ROI1), unless multiple distinct values are found.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Sort keys numerically
            keys = sorted([k for k in f.keys() if k.isdigit()], key=int)
            if not keys:
                print(f"  No numeric keys found in {h5_path}")
                return None
            
            # Determine last frame to size arrays (assuming continuous 0 to N)
            # Or just append? Plotter.create_pandas logic relies on arrays being same length.
            # We will just collect values and assume they correspond to frame 0, 1, 2...
            # But keys might be sparse? keys are frame numbers.
            # Plotter uses `range(len(it['x']))` which implies dense 0..N indices in the x array.
            # So if keys are [0, 1, 5], we probably need to fill gaps or specific handling.
            # However, looking at Plotter again:
            # df['ROI1.x'] = it['x']
            # So the length of it['x'] is the number of rows.
            # If we just append the found frames, the CSV rows will correspond to those found frames in order.
            # Ideally we should probably respect frame numbers, but `Plotter` doesn't take frame indices as input, 
            # it assumes the list IS the time series.
            # Let's stick strictly to what the user asked: "using castle/ui/plot_mask_info.py regenerate csv".
            # So we prepare `results` list.
            
            x_vals = []
            y_vals = []
            areas = []
            
            # Assuming one ROI for now.
            # If we need multi-ROI support, we'd need to check unique values in masks.
            
            for k in keys:
                data = f[k][:]
                # data is mask.
                
                # Check for unique ROI IDs if we want to be fancy, but let's assume binary/single mask for now.
                # If data has values 1, 2 etc, those are different ROIs.
                # Just doing value > 0 for now as per previous script logic.
                
                y_indices, x_indices = np.where(data > 0)
                
                if len(x_indices) == 0:
                    x_vals.append(0)
                    y_vals.append(0)
                    areas.append(0)
                else:
                    x_vals.append(np.mean(x_indices))
                    y_vals.append(np.mean(y_indices))
                    areas.append(len(x_indices))
            
            # Construct result item for ROI1
            item = {
                'x': np.array(x_vals),
                'y': np.array(y_vals),
                'area': np.array(areas)
            }
            
            return [item] # List of dicts, one per ROI
            
    except Exception as e:
        print(f"  Error processing {h5_path}: {e}")
        return None

def main():
    pattern = os.path.join(TRACK_DIR, '**', 'mask_list.h5')
    files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(files)} mask_list.h5 files.")
    
    for file_path in sorted(files):
        video_dir = os.path.dirname(file_path)
        video_name = os.path.basename(video_dir)
        # video_name usually 'S21.mp4'
        
        # Output CSV name: [VideoName]-basic-infomation.csv
        # Note: 'infomation' typo is present in existing files, preserving it.
        csv_name = f"{video_name.replace('.mp4', '')}-basic-infomation.csv"
        # Wait, previous `ls` showed `S21-basic-infomation.csv` inside `S21.mp4` folder.
        # And video dir name is `S21.mp4`.
        # So if video_name is `S21.mp4`, `video_name.replace('.mp4', '')` is `S21`.
        # Correct.
        
        output_csv = os.path.join(video_dir, csv_name)
        
        print(f"Processing {video_name} -> {output_csv}")
        
        results = get_roi_data(file_path)
        
        if results:
            try:
                df = Plotter.create_pandas(results)
                df.to_csv(output_csv, index=False)
                print(f"  Saved {output_csv}")
            except Exception as e:
                print(f"  Error generating CSV for {video_name}: {e}")
        else:
            print(f"  No results for {video_name}")

if __name__ == '__main__':
    main()
