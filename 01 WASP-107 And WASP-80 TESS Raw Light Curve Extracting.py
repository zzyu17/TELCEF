import lightkurve as lk
import os
import matplotlib.pyplot as plt

# Define the directories
root = os.getcwd()
search_result_dir = root + "/01 Search Result"
os.makedirs(search_result_dir, exist_ok=True)
raw_lightcurve_plots_dir = root + "/01 Raw Lightcurve Plots"
os.makedirs(raw_lightcurve_plots_dir, exist_ok=True)
wasp_107_raw_lightcurve_plots_dir = raw_lightcurve_plots_dir + "/WASP-107"
os.makedirs(wasp_107_raw_lightcurve_plots_dir, exist_ok=True)
wasp_80_raw_lightcurve_plots_dir = raw_lightcurve_plots_dir + "/WASP-80"
os.makedirs(wasp_80_raw_lightcurve_plots_dir, exist_ok=True)




# Search for all the available data of WASP-107 and WASP-80
wasp_107_lc_search_result = lk.search_lightcurve("WASP-107")
wasp_107_lc_search_result_file = open(search_result_dir+"/WASP-107 Lightcurve Search Result.txt", "w")
wasp_107_lc_search_result_file.write(str(wasp_107_lc_search_result))

wasp_80_lc_search_result = lk.search_lightcurve("WASP-80")
wasp_80_lc_search_result_file = open(search_result_dir+"/WASP-80 Lightcurve Search Result.txt", "w")
wasp_80_lc_search_result_file.write(str(wasp_80_lc_search_result))



# Download the K2 data
# wasp_107_lc_K2 = wasp_107_lc_search_result[(wasp_107_lc_search_result.mission >= 'K2') & (wasp_107_lc_search_result.mission != 'TESS') & (wasp_107_lc_search_result.author != 'K2SC') & (wasp_107_lc_search_result.author != 'DIAMANTE')].download_all()
# wasp_107_lc_K2_plot, ax1 = plt.subplots(figsize=(20,5))
# for lc in wasp_107_lc_K2:
#   # Normalize and plot the raw lightcurve collection
#   normalized_lc = lc.normalize()
#   wasp_107_lc_K2_plot = normalized_lc.plot(ax=ax1, label=f'{lc.mission} Pipeline {lc.author}')
#
#   # Plot the single raw lightcurve
#   wasp_107_lc_K2_plot_single , ax2 = plt.subplots(figsize=(20,5))
#   lc.plot(ax=ax2, label=f'{lc.mission} Pipeline {lc.author}')
#   wasp_107_lc_K2_plot_single.figure.tight_layout()
#   wasp_107_lc_K2_plot_single.figure.savefig(wasp_107_raw_lightcurve_plots_dir + f'/WASP-107 K2 Pipeline {lc.author} Raw Lightcurve.png')
# wasp_107_lc_K2_plot.figure.tight_layout()
# wasp_107_lc_K2_plot.figure.savefig(wasp_107_raw_lightcurve_plots_dir + "/WASP-107 K2 Raw Lightcurve.png")


# Download and plot the TESS data
wasp_107_lc_TESS = wasp_107_lc_search_result[(wasp_107_lc_search_result.mission >= 'TESS') & (wasp_107_lc_search_result.author != 'DIAMANTE') & (wasp_107_lc_search_result.author != 'GSFC-ELEANOR-LITE')].download_all()
wasp_107_lc_TESS_plot, ax1 = plt.subplots(figsize=(20,5))
for lc in wasp_107_lc_TESS:
  # Normalize and plot the raw lightcurve collection
  normalized_lc = lc.normalize()
  wasp_107_lc_TESS_plot = normalized_lc.plot(ax=ax1, label=f'{lc.mission} Sector {lc.sector} Pipeline {lc.author}')

  # Plot the single raw lightcurve
  wasp_107_lc_TESS_plot_single , ax2 = plt.subplots(figsize=(20,5))
  lc.plot(ax=ax2, label=f'{lc.mission} Sector {lc.sector} Pipeline {lc.author}')
  wasp_107_lc_TESS_plot_single.figure.tight_layout()
  wasp_107_lc_TESS_plot_single.figure.savefig(wasp_107_raw_lightcurve_plots_dir + f'/WASP-107 TESS Sector {lc.sector} Pipeline {lc.author} Raw Lightcurve .png')
wasp_107_lc_TESS_plot.figure.tight_layout()
wasp_107_lc_TESS_plot.figure.savefig(wasp_107_raw_lightcurve_plots_dir + "/WASP-107 TESS Raw Lightcurve.png")


wasp_80_lc_TESS = wasp_80_lc_search_result[(wasp_80_lc_search_result.mission >= 'TESS') & (wasp_80_lc_search_result.author != 'DIAMANTE')].download_all()
wasp_80_lc_TESS_plot, ax1 = plt.subplots(figsize=(20,5))
for lc in wasp_80_lc_TESS:
  # Normalize and plot the raw lightcurve collection
  normalized_lc = lc.normalize()
  wasp_80_lc_TESS_plot = normalized_lc.plot(ax=ax1, label=f'{lc.mission} Sector {lc.sector} Pipeline {lc.author}')

  # Plot the single raw lightcurve
  wasp_80_lc_TESS_plot_single , ax2 = plt.subplots(figsize=(20,5))
  lc.plot(ax=ax2, label=f'{lc.mission} Sector {lc.sector} Pipeline {lc.author}')
  wasp_80_lc_TESS_plot_single.figure.tight_layout()
  wasp_80_lc_TESS_plot_single.figure.savefig(wasp_80_raw_lightcurve_plots_dir + f'/WASP-80 TESS Sector {lc.sector} Pipeline {lc.author} Raw Lightcurve .png')
wasp_80_lc_TESS_plot.figure.tight_layout()
wasp_80_lc_TESS_plot.figure.savefig(wasp_80_raw_lightcurve_plots_dir + "/WASP-80 TESS Raw Lightcurve.png")