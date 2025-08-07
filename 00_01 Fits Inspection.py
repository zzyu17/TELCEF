import os
from astropy.io import fits




# Set the single sector and exposure time to be processed
sector = 54 ##### set the single sector to be processed #####
exptime = 600 ##### set the exposure time of data to be processed #####

# Define the starting point of TESS observation time
TESS_time = 2457000 # BJD


##### Define the source parameters #####
name = "WASP-80"
tic = 243921117
coords = (303.16737208333325,-2.144218611111111)
gaia = 4223507222112425344
tess_mag = 10.3622




# Define the directories
root = os.getcwd()
eleanor_root = os.path.expanduser("~/.eleanor")
eleanor_root_metadata = eleanor_root + f"/metadata"
eleanor_root_metadata_sector = eleanor_root_metadata + f"/s00{sector}"
eleanor_root_tesscut = eleanor_root + f"/tesscut"
eleanor_root_tesscut_target = eleanor_root_tesscut + f"/{name}"
fits_inspection_dir = root + "/00_01 Fits Inspection"
os.makedirs(fits_inspection_dir, exist_ok=True)




# Define the inspect_fits_file() function
def inspect_fits_file(fits_fn, fits_inspection_fn=None):
    """
    Function to inspect a FITS file and write its metadata into a text file.
    """
    # Define the filenames
    fits_fn_pure = os.path.basename(fits_fn).split(".")[0]
    if fits_inspection_fn is None:
        fits_inspection_fn = fits_inspection_dir + f"/{fits_fn_pure}.txt"
    fits_inspection_file = open(fits_inspection_fn, "w", encoding='utf-8')


    hdulist = fits.open(fits_fn)

    # Print the overall FITS information to the file
    hdulist.info(output=fits_inspection_file)
    fits_inspection_file.write("\n\n")

    # Print the header of the primary HDU to the file
    primary_hdu = hdulist[0]
    primary_header = primary_hdu.header
    fits_inspection_file.write("Primary Header:\n")
    for key, value in primary_header.items():
        comment = primary_header.comments[key]
        fits_inspection_file.write(f"{key} = {value} / {comment}\n")
    fits_inspection_file.write("\n")

    hdulist.close()
    fits_inspection_file.close()




# Inspect each FITS file in the assigned directory
for root, dirs, files in os.walk(eleanor_root_tesscut): ##### set the directory to be inspected #####
    for file in files:
        if file.endswith(".fits"):
            fits_fn = root + '/' + file
            print(f"Inspecting {fits_fn}...")
            inspect_fits_file(fits_fn)