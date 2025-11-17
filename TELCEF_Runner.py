import argparse
import time

from utils import run_script, load_config




# Parse the command-line argument to get the runner configuration filename
parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

# Load the runner configurations
runner_config_path = args.config if args.config is not None else "TELCEF_Runner_Configurations.yml" ##### set the runner configuration file path #####
runner_config = load_config(runner_config_path)


# Define the runner settings
stop_on_failure = runner_config['settings']['stop_on_failure'] if runner_config['settings']['stop_on_failure'] is not None else False
max_retries = runner_config['settings']['max_retries'] if runner_config['settings']['max_retries'] is not None else 0
retry_delay = runner_config['settings']['retry_delay'] if runner_config['settings']['retry_delay'] is not None else 5.0


n_sources = len(runner_config['sources'])
n_completed = 0
n_failed = 0
completed_sources = []
failed_sources = []




# Start the TELCEF Runner
print(f"Starting TELCEF Runner with {n_sources} source(s) to process...\n")

telcef_runner_total_start_time = time.time() # measure the start time


for source_name, source_config in runner_config['sources'].items():
    print(f"{'=' * 100}")
    print(f"Processing: {source_name} ({n_completed + 1}/{n_sources})")
    print(f"{'=' * 100}\n")

    # Run the scripts for the current source
    telcef_runner_single_start_time = time.time() # measure the start time

    success = True
    for script in source_config['scripts']:
        # Run the script with the specified configuration filename argument
        args = ["--config", source_config['config_fn']]
        if not run_script(script, args, max_retries, retry_delay):
            success = False
            break

    telcef_runner_single_end_time = time.time() # measure the end time
    telcef_runner_single_run_time = telcef_runner_single_end_time - telcef_runner_single_start_time # calculate the single run time

    if success:
        n_completed += 1
        completed_sources.append(source_name)
        print(f"\u2713 Completed processing {source_name} in {telcef_runner_single_run_time:.3f} seconds.\n\n")
    else:
        n_failed += 1
        failed_sources.append(source_name)

        if stop_on_failure == True:
            print(f"\u2715 Failed processing {source_name} after {telcef_runner_single_run_time:.3f} seconds.")
            print("Stopped due to failure.\n\n")
            break
        else:
            print(f"\u2715 Failed processing {source_name} after {telcef_runner_single_run_time:.3f} seconds.\n\n")


telcef_runner_total_end_time = time.time() # measure the end time
telcef_runner_total_run_time = telcef_runner_total_end_time - telcef_runner_total_start_time # calculate the total run time
telcef_runner_average_run_time = telcef_runner_total_run_time / n_sources # calculate the average run time

print(f"{'#' * 100}")
if failed_sources:
    print(f"\u2713 {n_completed}/{n_sources} source(s) processed successfully.")
    print(f"\u2715 Failed ({n_failed}/{n_sources}) source(s): {', '.join(failed_sources)}.\n")
else:
    print(f"\u2713 All {n_sources} source(s) processed successfully.\n")
print(f"Total run time: {telcef_runner_total_run_time:.3f} seconds.")
print(f"Average run time: {telcef_runner_average_run_time:.3f} seconds per source.")
print(f"{'#' * 100}\n")