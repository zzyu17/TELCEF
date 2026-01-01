import argparse
import os
import time
import multiprocessing as mp

from utils import run_source_worker, load_config




def main():
    # Parse the command-line argument to get the runner configuration filename
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--n_processes")
    args = parser.parse_args()

    # Load the runner configurations
    runner_config_path = args.config if args.config is not None else "TELCEF_Runner_Configurations.yml" ##### set the runner configuration file path #####
    runner_config = load_config(runner_config_path)


    # Load the sources configurations
    sources_name, sources_config = zip(*(runner_config["sources"].items()))
    n_sources = len(runner_config['sources'])
    n_completed = 0
    n_failed = 0
    completed_sources = []
    failed_sources = []


    # Define the runner settings
    multiprocessing = runner_config['settings']['multiprocessing']
    if runner_config['settings']['n_processes'] is None:
        n_processes = min(n_sources, int(0.8 * os.cpu_count()))
    else:
        n_processes = runner_config['settings']['n_processes']
    stop_on_failure = runner_config['settings']['stop_on_failure'] if runner_config['settings']['stop_on_failure'] is not None else False
    max_retries = runner_config['settings']['max_retries'] if runner_config['settings']['max_retries'] is not None else 0
    retry_delay = runner_config['settings']['retry_delay'] if runner_config['settings']['retry_delay'] is not None else 5.0




    # Start the TELCEF Runner
    print(f"Starting TELCEF Runner with {n_sources} source(s) to process...\n")

    telcef_runner_total_start_time = time.time() # measure the start time


    if not multiprocessing or n_processes == 1:
        # Sequential processing
        for i in range(n_sources):
            source_name = sources_name[i]
            source_config = sources_config[i]

            print(f"{'=' * 100}")
            print(f"Processing: {source_name} ({i + 1}/{n_sources})")
            print(f"{'=' * 100}\n")

            # Run source worker sequentially
            apply_results = run_source_worker(source_config, max_retries, retry_delay)
            [success, telcef_runner_single_run_time] = apply_results.values()

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


    else:
        # Parallel processing using multiprocessing
        with mp.Pool(processes=n_processes) as pool:
            sources_results = []

            for i in range(n_sources):
                source_name = sources_name[i]
                source_config = sources_config[i]

                print(f"{'=' * 100}")
                print(f"Processing: {source_name} ({i + 1}/{n_sources})")
                print(f"{'=' * 100}\n")

                # Apply async source worker
                apply_results = pool.apply_async(run_source_worker, args=(source_config, max_retries, retry_delay))
                sources_results.append({source_name: apply_results})

            pool.close()
            pool.join()

            # Collect workers results
            for source_results in sources_results:
                source_name = list(source_results.keys())[0]
                apply_results = source_results[source_name]
                [success, telcef_runner_single_run_time] = apply_results.get().values()

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




if __name__ == "__main__":
    main()