import os

from tess_stars2px import tess_stars2px_function_entry as tess_stars2px

from A_ab_Configuration_Loader import *




# Define the directories
tess_stars2px_inspection_dir = base_dir + config['directory']['tess_stars2px_inspection_dir']
os.makedirs(tess_stars2px_inspection_dir, exist_ok=True)
tess_stars2px_inspection_fn = tess_stars2px_inspection_dir + f"/tess_stars2px_{name}_Sector-{sector}.txt"




tess_stars2px_attribute = ['TIC', 'EclipticLong', 'EclipticLat', 'Sector', 'Camera', 'Ccd', 'ColPix', 'RowPix']

# Run tess_stars2px on the source
result = tess_stars2px(tic, coords[0], coords[1], sector)
tess_stars2px_inspection_file = open(tess_stars2px_inspection_fn, "w", encoding='utf-8')
for i in range(len(result)-1):
    tess_stars2px_inspection_file.write(f"{tess_stars2px_attribute[i]}: {result[i][0]}\n")




print("Successfully ran the tess_stars2px command on the source.\n")