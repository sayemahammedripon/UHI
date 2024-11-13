# UHI classes

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from raster_masker import RasterMasker  # Import the custom RasterMasker module
import matplotlib.colors as mcolors

# Define the paths
lst_raster_path = "/content/output_files/land_surface_temperature.tif"
shapefile_path = "/content/drive/MyDrive/Delhi/Delhi_UTM.shp"
output_uhi = "/content/output_files/uhi_class_intermediate.tif"
masked_output_uhi = "/content/output_files/uhi_class.tif"

try:
    # Open the LST raster file
    with rasterio.open(lst_raster_path) as src:
        lst_data = src.read(1).astype(float)
        original_nodata = src.nodata

        # Replace the original nodata value with np.nan
        lst_data[lst_data == original_nodata] = np.nan

        # Calculate the statistics
        mean_val = np.nanmean(lst_data)
        std_val = np.nanstd(lst_data)

        # Define UHI thresholds
        threshold_high = mean_val + 0.5 * std_val
        threshold_low = 0

        # Classify UHI areas
        uhi_high = (lst_data > threshold_high).astype(np.uint8)
        uhi_low = ((lst_data > threshold_low) & (lst_data <= threshold_high)).astype(np.uint8)

        # Combine the results into a single array (2: UHI, 1: Non-UHI, 0: Other)
        uhi_classification = np.zeros_like(lst_data, dtype=np.uint8)
        uhi_classification[uhi_low == 1] = 1
        uhi_classification[uhi_high == 1] = 2

        # Update profile for the output file
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        # Save the UHI classification raster
        with rasterio.open(output_uhi, 'w', **profile) as dst:
            dst.write(uhi_classification, 1)

    # Apply masking to the final UHI classification output
    masker = RasterMasker(shapefile_path)
    masker.mask_raster(output_uhi, masked_output_uhi)

    # Display the masked UHI classification
    with rasterio.open(masked_output_uhi) as masked_src:
        masked_uhi = masked_src.read(1)

        # Create a discrete colormap
        cmap = mcolors.ListedColormap(['white', 'yellow', 'red'])
        bounds = [0, 1, 2, 3]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 10))
        img = plt.imshow(masked_uhi, cmap=cmap, norm=norm)
        plt.title("UHI Classification (2: UHI, 1: Non-UHI, 0: Other)")

        # Add colorbar with ticks for each class
        cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5])
        cbar.ax.set_yticklabels(['0: Other', '1: Non-UHI', '2: UHI'])
        cbar.set_label("Classification")

        plt.show()

    print(f"UHI classification saved as: {masked_output_uhi}")

except Exception as e:
    print(f"Error occurred: {e}")
