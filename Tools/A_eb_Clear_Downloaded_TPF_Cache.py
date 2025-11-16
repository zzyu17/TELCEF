from utils import remove_subfolder, remove_zip
from A_ab_Configuration_Loader import *




# Define the directories
tpf_downloaded_dir = data_dir + config['directory']['tpf_downloaded_dir']
os.makedirs(tpf_downloaded_dir, exist_ok=True)
tpf_downloaded_dir_source = tpf_downloaded_dir + f"/{name}"
os.makedirs(tpf_downloaded_dir_source, exist_ok=True)




remove_subfolder(tpf_downloaded_dir_source)
remove_zip(tpf_downloaded_dir_source)




print("Successfully cleared downloaded TPF cache.")