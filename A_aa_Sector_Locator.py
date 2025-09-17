import os
import subprocess

import lightkurve as lk
import numpy as np

from utils import load_config, update_config




# Define the base directory
base_dir = os.path.dirname(__file__) ##### set the base directory of input and output #####
# Load the configurations and set the "config" global variable
config_dir = base_dir + "/Configurations"
config_fn = "/WASP-80_b_Configurations_and_Results.yml" ##### set the configuration filename based on the source star and planet name #####
config_path = config_dir + config_fn
config = load_config(config_path)




##### Define the source parameters #####
name = config['source']['name'] ##### define the source name #####




if __name__ == "__main__":
    # Run the tess_stars2px command to locate the sector, camera, CCD and CCD coordinates for the source
    result = subprocess.run(["python", "-m", "tess_stars2px", "-n", name], capture_output=True, text=True)

    # parse the result
    result_lines = result.stdout.strip().split("\n")
    parsed_result_list = []
    for line in result_lines[2:]:  # Assuming the first two lines are headers
        parsed_result_list.append([float(x) if '.' in x else int(x) for x in line.split("|")])
    parsed_result_array = np.array(parsed_result_list)

    # extract and print the sector, camera, CCD and CCD coordinates information
    sector_num = parsed_result_array.shape[0]
    sectors = [int(parsed_result_array[i, 5]) for i in range(sector_num)]
    cameras = [int(parsed_result_array[i, 6]) for i in range(sector_num)]
    ccds = [int(parsed_result_array[i, 7]) for i in range(sector_num)]
    ccd_coords = [(float(parsed_result_array[i, 8]), float(parsed_result_array[i, 9])) for i in range(sector_num)]
    sccs = [[sector, camera, ccd, ccd_coord] for sector, camera, ccd, ccd_coord in zip(sectors, cameras, ccds, ccd_coords)] # stack the sector-camera-CCD-CCD_coordinates combinations into a list


    # Utilize the lightkurve.search_targetpixelfile() to find the available TPF exptime(s) for the source in the located sectors
    tpf_exptimes = []
    for scc in sccs:
        tpf_search_result = lk.search_targetpixelfile(name, mission=config['mission']['mission'], sector=scc[0])
        tpf_exptimes.append(sorted(set(int(tpf_exptime) for tpf_exptime in tpf_search_result.exptime.value)))


    # Utilize the lightkurve.search_lightcurve() to find the available lightcurve exptime(s) for the source in the located sectors
    lc_exptimes = []
    for scc in sccs:
        lc_search_result = lk.search_lightcurve(name, mission=config['mission']['mission'], sector=scc[0])
        lc_exptimes.append(sorted(set(int(lc_exptime) for lc_exptime in lc_search_result.exptime.value)))


    print(f"Collected the SCCE (sector, camera, CCD, CCD coordinates and exptime) information for {name}.\n")


    # Update the configurations
    config['source']['available_sectors'] = sectors
    config['source']['available_tpf_exptimes'] = tpf_exptimes
    config['source']['available_lc_exptimes'] = lc_exptimes
    dict_update = {'source.available_sectors': sectors,
                   'source.available_tpf_exptimes': tpf_exptimes,
                   'source.available_lc_exptimes': lc_exptimes}
    config = update_config(config_path, dict_update)


    # Write the sector, camera, CCD, CCD coordinates and exptime information to a text file
    scce_path = config_dir + f"/{name}_SCCE_Info.txt"
    scce_file = open(scce_path, "w", encoding='utf-8')

    scce_file.write(f"Found {name} in {sector_num} sector(s), with the following sector, camera, CCD, CCD coordinates and exptime information:\n")
    for i in range(sector_num):
        scce_file.write(f"sector {sectors[i]}, camera {cameras[i]}, CCD {ccds[i]}, CCD coordinates {ccd_coords[i]} and available exptime(s) for the source in this sector: ")
        scce_file.write(f"TPF exptime(s) includes " + ', '.join([str(int(tpf_exptime)) + "s" for tpf_exptime in tpf_exptimes[i]]) + "; ")
        scce_file.write(f"lightcurve exptime(s) includes " + ', '.join([str(int(lc_exptime)) + "s" for lc_exptime in lc_exptimes[i]]) + ".\n")
    scce_file.close()