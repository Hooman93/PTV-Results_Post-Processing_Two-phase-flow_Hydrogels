# ====================================================================
# COMPUTE AND COMPARE DISPERSION COEFFICIENTS FROM FOUR STB DATASETS
# ====================================================================
#
# For FOUR Shake-the-Box (STB) data folders:
# - Reads 3D particle tracks (x, y, z over time)
# - Computes axial (Y) and radial (X-Z) Mean Squared Displacement (MSD)
# - Averages MSD over all tracks for time intervals up to 2 seconds
# - Defines reported dispersion coefficient as the fit over last 0.2 s
#
# Finally:
# - Plots all RADIAL MSD curves together (in mm²)
# - Plots all AXIAL MSD curves together (in mm²)
# - Plots DISPERSION COEFFICIENT vs Solid Volume Fraction (in mm²/s)
#
# Author: ChatGPT
# ====================================================================

import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

# Imaging frame rate (Hz)
# ------------------------------------------------------------
fps = 40.0
time_step = 1.0 / fps  # e.g., 0.025 seconds between frames
print(f"Frame interval between images: {time_step:.4f} seconds")

# Analysis time window
# ------------------------------------------------------------
analysis_window_seconds = 2.0
max_time_intervals = int(analysis_window_seconds / time_step)
print(f"Number of time intervals to analyze: {max_time_intervals}")

# ------------------------------------------------------------
# Paths to FOUR data folders with .set files from DaVis
# ------------------------------------------------------------
data_paths = [
    r"C:\python files\TWO-PHASE_HYDROGELS\track data\half top\Dispersion analysis_0%",
    r"C:\python files\TWO-PHASE_HYDROGELS\track data\half top\Dispersion analysis_10%",
    r"C:\python files\TWO-PHASE_HYDROGELS\track data\half top\Dispersion analysis_30%",
    r"C:\python files\TWO-PHASE_HYDROGELS\track data\half top\Dispersion analysis_50%",
]

# Labels for plots and reporting
labels = [
    "Case 1: 0% solids",
    "Case 2: 10% solids",
    "Case 3: 30% solids",
    "Case 4: 50% solids"
]

# Duration of final fit region for dispersion coefficient reporting
fit_window_duration = 0.2  # seconds

# ------------------------------------------------------------
# Storage for all results
# ------------------------------------------------------------
all_time_intervals_sec = []    # x-axis for MSD plots
all_msd_axial = []             # Axial MSD curves for each case
all_msd_radial = []            # Radial MSD curves for each case
reported_D_axial = []          # Single dispersion coefficient per case (axial)
reported_D_radial = []         # Single dispersion coefficient per case (radial)

# ------------------------------------------------------------
# MAIN LOOP OVER ALL DATASETS
# ------------------------------------------------------------
for idx, path in enumerate(data_paths):

    print("\n-------------------------------------------------")
    print(f"Processing {labels[idx]}")
    print(f"Data folder: {path}")

    # --------------------------------------------------------
    # Load particle tracks from .set folder
    # --------------------------------------------------------
    tr = lv.read_particles(path)
    tracks = tr.tracks()
    num_tracks = len(tracks)
    print(f"Number of tracks found: {num_tracks}")

    if num_tracks == 0:
        print("No tracks found in this dataset. Skipping.")
        continue

    # --------------------------------------------------------
    # Initialize arrays to accumulate MSD sums
    # --------------------------------------------------------
    msd_axial_sum = np.zeros(max_time_intervals)
    msd_radial_sum = np.zeros(max_time_intervals)
    counts = np.zeros(max_time_intervals)

    # --------------------------------------------------------
    # Process each track to compute squared displacements
    # --------------------------------------------------------
    for track in tracks:
        positions = track.particles
        x = positions["x"]
        y = positions["y"]
        z = positions["z"]
        num_points = len(x)

        if num_points <= 2:
            continue  # too short to analyze

        for interval in range(1, min(max_time_intervals, num_points)):
            displacement_axial = (y[interval:] - y[:-interval])**2
            displacement_radial = (x[interval:] - x[:-interval])**2 + (z[interval:] - z[:-interval])**2

            msd_axial_sum[interval] += np.sum(displacement_axial)
            msd_radial_sum[interval] += np.sum(displacement_radial)
            counts[interval] += len(displacement_axial)

    # --------------------------------------------------------
    # Compute ensemble-averaged MSD over all tracks
    # --------------------------------------------------------
    valid = counts > 0
    msd_axial = np.zeros_like(msd_axial_sum)
    msd_radial = np.zeros_like(msd_radial_sum)
    msd_axial[valid] = msd_axial_sum[valid] / counts[valid]
    msd_radial[valid] = msd_radial_sum[valid] / counts[valid]

    # Store time intervals in seconds for x-axis
    time_intervals_sec = np.arange(max_time_intervals) * time_step
    all_time_intervals_sec.append(time_intervals_sec)
    all_msd_axial.append(msd_axial)
    all_msd_radial.append(msd_radial)

    # --------------------------------------------------------
    # LINEAR FIT IN LAST 0.2 SECONDS REGION
    # --------------------------------------------------------
    # MSD ≈ 2D Δt + c => D = slope / 2
    fit_end = time_intervals_sec[-1]
    fit_start = max(0, fit_end - fit_window_duration)
    fit_mask = (time_intervals_sec >= fit_start) & (time_intervals_sec <= fit_end)

    if np.sum(fit_mask) >= 2:
        coeffs_axial = np.polyfit(time_intervals_sec[fit_mask], msd_axial[fit_mask], 1)
        coeffs_radial = np.polyfit(time_intervals_sec[fit_mask], msd_radial[fit_mask], 1)

        slope_axial = coeffs_axial[0]
        slope_radial = coeffs_radial[0]

        D_axial_reported = slope_axial / 2
        D_radial_reported = slope_radial / 2

        reported_D_axial.append(D_axial_reported)
        reported_D_radial.append(D_radial_reported)

        print(f"✅ Reported Axial Dispersion Coefficient (last 0.2s): {D_axial_reported:.4e} mm²/s")
        print(f"✅ Reported Radial Dispersion Coefficient (last 0.2s): {D_radial_reported:.4e} mm²/s")
    else:
        reported_D_axial.append(np.nan)
        reported_D_radial.append(np.nan)
        print("⚠️ Warning: Not enough points in last 0.2s to fit.")

# ------------------------------------------------------------
# PLOTTING RESULTS
# ------------------------------------------------------------
# All datasets now processed. Plot results for comparison.

# ------------------------------------------------------------
# Plot all RADIAL MSD curves
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
for i in range(len(all_msd_radial)):
    plt.plot(all_time_intervals_sec[i], all_msd_radial[i], label=labels[i])
plt.xlabel("Time Interval (seconds)", fontsize=12)
plt.ylabel("Radial Mean Squared Displacement (mm²)", fontsize=12)
plt.title("Radial MSD vs Time Interval (Comparison of All Cases)", fontsize=13)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Plot all AXIAL MSD curves
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
for i in range(len(all_msd_axial)):
    plt.plot(all_time_intervals_sec[i], all_msd_axial[i], label=labels[i])
plt.xlabel("Time Interval (seconds)", fontsize=12)
plt.ylabel("Axial Mean Squared Displacement (mm²)", fontsize=12)
plt.title("Axial MSD vs Time Interval (Comparison of All Cases)", fontsize=13)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# REPORT FINAL DISPERSION COEFFICIENTS
# ------------------------------------------------------------
print("\n✅ FINAL REPORTED DISPERSION COEFFICIENTS (from last 0.2 seconds):\n")
for i in range(len(labels)):
    print(f"{labels[i]}:")
    print(f"  Axial D  = {reported_D_axial[i]:.4e} mm²/s")
    print(f"  Radial D = {reported_D_radial[i]:.4e} mm²/s")
    print()

# ------------------------------------------------------------
# PLOT: Dispersion Coefficient vs Solid Volume Fraction
# ------------------------------------------------------------
solid_vof = [0, 10, 30, 50]

plt.figure(figsize=(7, 5))
plt.plot(solid_vof, reported_D_axial, 'bo-', label='Axial Dispersion Coefficient')
plt.plot(solid_vof, reported_D_radial, 'ro-', label='Radial Dispersion Coefficient')
plt.xlabel("Solid Volume Fraction (%)", fontsize=12)
plt.ylabel("Dispersion Coefficient (mm²/s)", fontsize=12)
plt.title("Dispersion Coefficient vs Solid Volume Fraction", fontsize=13)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

print("✅ ALL DONE!")
