print('''------ AMPM Project ------
*** Testing Model ***
''')

import torch
import os
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
from pathlib import Path
import glob
import re

import_script_start = datetime.now()
from use_model.arguments import initiate_argument
from use_model.read_log import read_log
from use_model.data_function import process_data
from interface_prediction.run_setup.set_up import create_output_folder
from interface_prediction.model.testing_function import test_model
from interface_prediction.evaluation_and_plot.boxplot import result_boxplot
from interface_prediction.evaluation_and_plot.result_evaluation import table_mean
from interface_prediction.model.model_function import choose_model
import_script_time = datetime.now() - import_script_start
print(f'Import necessary library: {import_script_time}')


def cleanup_old_logs(logs_dir: str = 'logs'):
    """
    Clean up old log files, keeping only the most recent log from previous day.
    
    For each job type (e.g., epi4ab-infer-all-*.out), keeps only the most recent
    log file from the previous day and deletes all older logs.
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return
    
    print("Cleaning up old log files...")
    
    # Get current date
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Find all log files
    log_files = list(logs_path.glob('*.out')) + list(logs_path.glob('*.err'))
    
    if not log_files:
        print("  No log files found.")
        return
    
    # Group logs by job type (extract job name pattern)
    job_patterns = {}
    for log_file in log_files:
        # Extract job name pattern (e.g., "epi4ab-infer-all" from "epi4ab-infer-all-12345.out")
        match = re.match(r'([^-]+(?:-[^-]+)*)-\d+\.(out|err)', log_file.name)
        if match:
            job_name = match.group(1)
            log_type = match.group(2)
            key = f"{job_name}.{log_type}"
            
            if key not in job_patterns:
                job_patterns[key] = []
            job_patterns[key].append(log_file)
    
    deleted_count = 0
    kept_count = 0
    
    for job_key, files in job_patterns.items():
        # Get modification dates
        files_with_dates = []
        for f in files:
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                files_with_dates.append((f, mtime.date(), mtime))
            except (FileNotFoundError, OSError) as e:
                # Skip files that were deleted or are inaccessible
                continue
        
        # Separate files by date
        today_files = [(f, dt, mt) for f, dt, mt in files_with_dates if dt == today]
        yesterday_files = [(f, dt, mt) for f, dt, mt in files_with_dates if dt == yesterday]
        older_files = [(f, dt, mt) for f, dt, mt in files_with_dates if dt < yesterday]
        
        # Keep all today's files
        kept_count += len(today_files)
        
        # Keep only most recent from yesterday
        if yesterday_files:
            yesterday_files.sort(key=lambda x: x[2], reverse=True)
            kept_count += 1
            # Delete other yesterday files
            for f, _, _ in yesterday_files[1:]:
                try:
                    f.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  Warning: Could not delete {f.name}: {e}")
        
        # Delete all older files
        for f, _, _ in older_files:
            try:
                f.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  Warning: Could not delete {f.name}: {e}")
    
    print(f"  Cleanup complete: Kept {kept_count} files, deleted {deleted_count} old files.")

# Clean up old log files before starting
cleanup_old_logs()

# Initiate arguments
prepare_start_time = datetime.now()
args = initiate_argument()

# Create output folder
if os.path.isdir(args.directory_output) == False:
    os.mkdir(args.directory_output)

# Read model's log.md
logging = read_log(args)
setup_logging_time = datetime.now() - prepare_start_time
print(f'Set up logging: {setup_logging_time}')

# Load model
load_model_start_time = datetime.now()
if logging.save_not_as_statedict:
    model = torch.load(os.path.join(logging.directory_model_folder, 'model.pt'))
else:
    model = choose_model(logging).to(logging.device)
    model.load_state_dict(torch.load(os.path.join(logging.directory_model_folder, 'model.pt'), weights_only=True))
model.to(logging.device)
load_model_time = datetime.now() - load_model_start_time
print(f'Loading model: {load_model_time}')

# Create run session output folder
folder_name = str(logging.run_date) + '_' + logging.model_name
output_folder = create_output_folder(folder_name, logging.directory_output)
logging.directory_output_folder = output_folder
test_record_folder = os.path.join(output_folder,'test_record')
os.mkdir(test_record_folder)
logging.directory_test_record = test_record_folder
print(f'Results are located in {output_folder}.')
logging.prepare_time = datetime.now() - prepare_start_time

# Process test data
data_start_time = datetime.now()
test_data, test_list = process_data(logging)
logging.data_time = datetime.now() - data_start_time

# Use model
test_start_time = datetime.now()
evaluation_record = test_model(model, test_data, test_list, logging, testType='test')
logging.test_time = datetime.now() - test_start_time

# Convert evaluation data to pandas
result_start_time = datetime.now()
evaluation_df = pd.DataFrame(evaluation_record, columns = logging.info_record_columns)

plotting_boxplot_start_time = datetime.now()
result_boxplot(evaluation_df, logging, resultRegion = logging.use_region)
plotting_boxplot_time = datetime.now() - plotting_boxplot_start_time
print(f'Plotting boxplot: {plotting_boxplot_time}')

generate_table_start_time = datetime.now()
table_mean(evaluation_df, logging, resultRegion = logging.use_region)
generate_table_time = datetime.now() - generate_table_start_time
print(f'Summarize average result: {generate_table_time}')


with open(os.path.join(output_folder, 'evaluation_each_pdb.txt'), 'w') as f:
        f.write(tabulate(evaluation_record, headers=logging.info_record_columns))
        f.write('\n')
logging.result_time = datetime.now()

# Generate interactive HTML visualizations
visualizer_start_time = datetime.now()
try:
    print('Generating interactive HTML visualizations...', end=' ')
    from generate_visualizer import generate_visualizations
    
    eval_file = os.path.join(output_folder, 'evaluation_each_pdb.txt')
    eval_file_path = eval_file if os.path.exists(eval_file) else None
    
    generate_visualizations(
        test_record_dir=test_record_folder,
        output_dir=output_folder,
        pdb_id=None,  # Generate for all PDBs
        evaluation_file=eval_file_path,
        pdb_files_dir='/leonardo_scratch/fast/AIFAC_F01_302/epi4ab/pdb_files'
    )
    
    visualizer_time = datetime.now() - visualizer_start_time
    print(f'Done ({visualizer_time})')
    print(f'  Visualizations saved to: {output_folder}')
except Exception as e:
    print(f'Warning: Could not generate visualizations: {e}')
    import traceback
    traceback.print_exc()

print('Generate log: ', end='')
logging.log_result()
print('Done')