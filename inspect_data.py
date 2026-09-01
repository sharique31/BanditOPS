import os
import glob
import h5py
import numpy as np


# ============================================================
# DATASET PATHS
# ============================================================

BASE_DIR = r"C:\SIH\CLAUDE\FILES\DATA"

SCAN_DIR = os.path.join(BASE_DIR, "scan", "test_scan")
STARE_DIR = os.path.join(BASE_DIR, "stare", "test_stare")


# ============================================================
# FUNCTION TO INSPECT ONE HDF5 FILE
# ============================================================

def inspect_file(file_path):
    print("\n" + "=" * 80)
    print("FILE:")
    print(file_path)
    print("=" * 80)

    with h5py.File(file_path, "r") as f:

        print("\nROOT KEYS:")
        print(list(f.keys()))

        for key in f.keys():

            obj = f[key]

            print("\n" + "-" * 60)
            print(f"KEY: {key}")
            print(f"TYPE: {type(obj)}")

            if isinstance(obj, h5py.Dataset):

                print(f"SHAPE: {obj.shape}")
                print(f"DTYPE: {obj.dtype}")

                # Print attributes
                if len(obj.attrs.keys()) > 0:
                    print("\nATTRIBUTES:")
                    for attr_key, attr_value in obj.attrs.items():
                        print(f"  {attr_key}: {attr_value}")

            elif isinstance(obj, h5py.Group):

                print("GROUP CONTENTS:")
                print(list(obj.keys()))

                if len(obj.attrs.keys()) > 0:
                    print("\nGROUP ATTRIBUTES:")
                    for attr_key, attr_value in obj.attrs.items():
                        print(f"  {attr_key}: {attr_value}")

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        if "metadata" in f:

            metadata = f["metadata"]

            print("\n" + "=" * 60)
            print("METADATA")
            print("=" * 60)

            if isinstance(metadata, h5py.Group):

                print("METADATA KEYS:")
                print(list(metadata.keys()))

                for key in metadata.keys():

                    try:
                        value = metadata[key][()]

                        print(f"\n{key}:")
                        print(value)

                    except Exception as e:
                        print(f"\n{key}: Could not read")
                        print(e)

            elif isinstance(metadata, h5py.Dataset):

                try:
                    print(metadata[()])
                except Exception as e:
                    print(e)

        # ----------------------------------------------------
        # FEATURE NAMES
        # ----------------------------------------------------

        if "metadata" in f:

            metadata = f["metadata"]

            if isinstance(metadata, h5py.Group):

                if "feature_names" in metadata:

                    feature_names = metadata["feature_names"][()]

                    print("\n" + "=" * 60)
                    print("FEATURE NAMES")
                    print("=" * 60)

                    decoded_names = []

                    for name in feature_names:

                        if isinstance(name, bytes):
                            name = name.decode("utf-8")

                        decoded_names.append(str(name))

                    print(decoded_names)

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        if "data" in f:

            data = f["data"]

            print("\n" + "=" * 60)
            print("DATA INFORMATION")
            print("=" * 60)

            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")

            print("\nFIRST 5 ROWS:")

            try:
                print(data[:5])
            except Exception as e:
                print("Could not print data.")
                print(e)

            # Statistics

            try:

                data_array = data[:]

                print("\nDATA STATISTICS:")

                print("\nMinimum:")
                print(np.min(data_array, axis=0))

                print("\nMaximum:")
                print(np.max(data_array, axis=0))

                print("\nMean:")
                print(np.mean(data_array, axis=0))

                print("\nStandard Deviation:")
                print(np.std(data_array, axis=0))

            except Exception as e:

                print("\nCould not calculate statistics.")
                print(e)

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        if "labels" in f:

            labels = f["labels"][:]

            unique_labels, counts = np.unique(
                labels,
                return_counts=True
            )

            print("\n" + "=" * 60)
            print("LABEL INFORMATION")
            print("=" * 60)

            print(f"Total labels: {len(labels)}")

            print("\nUNIQUE LABELS AND COUNTS:")

            for label, count in zip(unique_labels, counts):

                print(
                    f"Label {label}: {count} samples"
                )


# ============================================================
# FIND FILES
# ============================================================

scan_files = sorted(
    glob.glob(
        os.path.join(SCAN_DIR, "*.h5")
    )
)

stare_files = sorted(
    glob.glob(
        os.path.join(STARE_DIR, "*.h5")
    )
)


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

print("\n" + "#" * 80)
print("COMPLETE DATASET SUMMARY")
print("#" * 80)

print(f"\nSCAN directory:")
print(SCAN_DIR)

print(f"Number of SCAN files: {len(scan_files)}")


print(f"\nSTARE directory:")
print(STARE_DIR)

print(f"Number of STARE files: {len(stare_files)}")


print("\nTOTAL FILES:")
print(len(scan_files) + len(stare_files))


# ============================================================
# INSPECT MULTIPLE FILES
# ============================================================

print("\n" + "#" * 80)
print("INSPECTING SCAN FILES")
print("#" * 80)


if len(scan_files) > 0:

    files_to_check = [
        scan_files[0],
        scan_files[len(scan_files) // 2],
        scan_files[-1]
    ]

    for file_path in files_to_check:

        inspect_file(file_path)

else:

    print("NO SCAN FILES FOUND!")


print("\n" + "#" * 80)
print("INSPECTING STARE FILES")
print("#" * 80)


if len(stare_files) > 0:

    files_to_check = [
        stare_files[0],
        stare_files[len(stare_files) // 2],
        stare_files[-1]
    ]

    for file_path in files_to_check:

        inspect_file(file_path)

else:

    print("NO STARE FILES FOUND!")


print("\n" + "#" * 80)
print("INSPECTION COMPLETE")
print("#" * 80)