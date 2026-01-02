import shutil

from utils import run_script, update_config
from A_ab_Configuration_Loader import *




base_config_fn = f"/WD1856+534_b_Configurations_and_Results.yml" ##### set the base configuration filename #####
base_config_path = config_dir + base_config_fn
base_config = load_config(base_config_path)


max_retries = 2
retry_delay = 5.0

sectors = [sector for sector in base_config['source']['available_sectors'] if sector >= 40]
n_fns = len(sectors)
n_completed = 0
n_failed = 0
completed_fns = []
failed_fns = []

for sector in sectors:
    new_config_fn = f"/WD1856+534_b_Sector-{sector}_Author-SPOC_Configurations_and_Results.yml" ##### set the new configuration filename #####
    new_config_fn_base = new_config_fn.lstrip('/')
    new_config_path = config_dir + new_config_fn

    print(f"{'=' * 100}")
    print(f"Processing: {new_config_fn_base} ({n_completed + 1}/{n_fns})")
    print(f"{'=' * 100}\n")

    try:
        shutil.copy2(base_config_path, new_config_path)
        update_config(new_config_path, {'source.sector': sector}) ##### update the specific value(s) in the new configuration file #####
        success = run_script("A_ab_Configuration_Loader.py", ["--config", new_config_fn], max_retries, retry_delay)
    except Exception as e:
        success = False
        print(f"Error while processing {new_config_fn_base}: {e}")

    if success:
        n_completed += 1
        completed_fns.append(new_config_fn_base)
        print(f"\u2713 Completed processing {new_config_fn_base}.\n\n")
    else:
        n_failed += 1
        failed_fns.append(new_config_fn_base)
        print(f"\u2715 Failed processing {new_config_fn_base}.\n\n")

print(f"{'#' * 100}")
if failed_fns:
    print(f"\u2713 {n_completed}/{n_fns} file(s) processed successfully.")
    joined = ',\n'.join(failed_fns)
    print(f"\u2715 Failed ({n_failed}/{n_fns}) file(s):\n {joined}.")
else:
    print(f"\u2713 All {n_fns} file(s) processed successfully.")
print(f"{'#' * 100}\n")