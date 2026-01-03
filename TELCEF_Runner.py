import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from utils import run_source_worker, load_config




# Parse the command-line argument to get the runner configuration filename
parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

# Load the runner configurations
runner_config_path = args.config if args.config is not None else "TELCEF_Runner_Configurations.yml" ##### set the runner configuration file path #####
runner_config = load_config(runner_config_path)


# Load the sources configurations
sources_name, sources_config = zip(*(runner_config["sources"].items()))
n_sources = len(runner_config['sources'])
n_succeeded = 0
n_failed = 0
succeeded_sources = []
failed_sources = []


# Define the runner settings
multiprocessing = runner_config['settings']['multiprocessing'] if runner_config['settings']['multiprocessing'] is not None else False
stop_on_failure = runner_config['settings']['stop_on_failure'] if runner_config['settings']['stop_on_failure'] is not None else False
max_retries = runner_config['settings']['max_retries'] if runner_config['settings']['max_retries'] is not None else 0
retry_delay = runner_config['settings']['retry_delay'] if runner_config['settings']['retry_delay'] is not None else 5.0
close_terminal = runner_config['settings']['close_terminal'] if runner_config['settings']['close_terminal'] is not None else False




print(f"Starting TELCEF Runner with {n_sources} source(s) to process...\n")

telcef_runner_total_start_time = time.time()  # measure the start time


if not multiprocessing:
    # Sequential processing
    for source_index in range(n_sources):
        source_name = sources_name[source_index]
        source_config = sources_config[source_index]

        print(f"{'=' * 100}")
        print(f"Processing: {source_name} ({source_index + 1}/{n_sources})")
        print(f"{'=' * 100}\n")

        # Run source worker sequentially
        success = run_source_worker(source_config, max_retries, retry_delay, use_terminal=False, source_name=source_name, stop_on_failure=stop_on_failure)

        if success:
            n_succeeded += 1
            succeeded_sources.append(source_name)
        else:
            n_failed += 1
            failed_sources.append(source_name)


else:
    # Parallel processing using separate terminals
    print(f"Processing {n_sources} source(s) in parallel using separate terminals, please check the terminal windows for progress...")

    for source_index in range(n_sources):
        source_name = sources_name[source_index]
        source_config = sources_config[source_index]

        print(f"{'=' * 100}")
        print(f"Launching: {source_name} ({source_index + 1}/{n_sources})")
        print(f"{'=' * 100}\n")

        # Run source worker in parallel using separate terminals
        success = run_source_worker(source_config, max_retries=0, use_terminal=True, source_name=source_name, close_terminal=close_terminal)

        if success:
            n_succeeded += 1
            succeeded_sources.append(source_name)
        else:
            n_failed += 1
            failed_sources.append(source_name)


telcef_runner_total_end_time = time.time() # measure the end time
telcef_runner_total_run_time = telcef_runner_total_end_time - telcef_runner_total_start_time # calculate the total run time
telcef_runner_average_run_time = telcef_runner_total_run_time / n_sources # calculate the average run time

print(f"{'#' * 100}")
if failed_sources:
    print(f"\u2713 {n_succeeded}/{n_sources} source(s) processed/launched in terminal successfully.")
    print(f"\u2715 Failed ({n_failed}/{n_sources}) source(s): {', '.join(failed_sources)}.\n")
else:
    print(f"\u2713 All {n_sources} source(s) processed/launched in terminal successfully.\n")
print(f"Total run/launch time: {telcef_runner_total_run_time:.3f} seconds.")
print(f"Average run/launch time: {telcef_runner_average_run_time:.3f} seconds per source.")
print(f"{'#' * 100}\n")