from utils import remove_subfolder, remove_zip
from A_ab_Configuration_Loader import *




# Define the directories
lc_downloaded_dir = data_dir + config['directory']['lc_downloaded_dir']
os.makedirs(lc_downloaded_dir, exist_ok=True)
lc_downloaded_dir_source = lc_downloaded_dir + f"/{name}"
os.makedirs(lc_downloaded_dir_source, exist_ok=True)




remove_subfolder(lc_downloaded_dir_source)
remove_zip(lc_downloaded_dir_source)




print("Successfully cleared downloaded light curve cache.")