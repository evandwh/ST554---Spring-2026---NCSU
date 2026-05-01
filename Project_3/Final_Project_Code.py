#########################################################
#
# Author: Evan Whitfield
# Date Last Editted: 4/30/2026
# Purpose: ST 554 Spring 2026 - Final Project
#
##########################################################

import pandas as pd
import time
import os

# Load data
data = pd.read_csv("power_streaming_data.csv")

# Output folder
output_directory = "Final Project Output"

# Loop for sampling
for i in range(20):

    # Sample 5 rows
    sample = data.sample(n = 5)

    # Remove index
    sample = sample.reset_index(drop = True)

    # Write each batch to separate file
    file_path = os.path.join(output_directory, f"batch_{i}.csv")
    sample.to_csv(file_path, index = False)

    # Log Step
    print(f"Wrote batch {i} to {file_path}")

    # Wait for next sample (20 is said to do better than 10)
    time.sleep(20)