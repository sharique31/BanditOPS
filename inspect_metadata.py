import h5py
import os
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Change these paths ONLY if your actual folder structure differs.
FILES = [
    r"C:\SIH\CLAUDE\FILES\DATA\scan\test_scan\config_0.h5",
    r"C:\SIH\CLAUDE\FILES\DATA\scan\test_scan\config_99.h5",
    r"C:\SIH\CLAUDE\FILES\DATA\scan\test_scan\config_210.h5",

    r"C:\SIH\CLAUDE\FILES\DATA\stare\test_stare\config_0.h5",
    r"C:\SIH\CLAUDE\FILES\DATA\stare\test_stare\config_99.h5",
    r"C:\SIH\CLAUDE\FILES\DATA\stare\test_stare\config_210.h5",
]

# Save output in the same folder where this Python script is located
OUTPUT_FILE = Path(__file__).parent / "metadata_output.txt"

# Maximum number of dataset values to print
MAX_VALUES_TO_PRINT = 50


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_value(value):
    """
    Convert bytes values into readable strings where possible.
    """

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return str(value)

    return value


def format_value(value):
    """
    Make HDF5 values easier to read.
    """

    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass

    if isinstance(value, list):

        def decode_recursive(item):
            if isinstance(item, list):
                return [decode_recursive(x) for x in item]

            return decode_value(item)

        value = decode_recursive(value)

    else:
        value = decode_value(value)

    return value


def print_attributes(obj, indent=0):
    """
    Print all attributes belonging to an HDF5 object.
    """

    prefix = " " * indent

    if len(obj.attrs) == 0:
        return

    print(f"{prefix}ATTRIBUTES:")

    for key, value in obj.attrs.items():
        print(f"{prefix}  {key}: {format_value(value)}")


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(name, dataset, indent=0):
    """
    Inspect an HDF5 dataset.
    """

    prefix = " " * indent

    print(f"{prefix}DATASET: {name}")
    print(f"{prefix}  Shape: {dataset.shape}")
    print(f"{prefix}  Dtype: {dataset.dtype}")

    print_attributes(dataset, indent + 2)

    try:
        total_elements = dataset.size

        if total_elements <= MAX_VALUES_TO_PRINT:

            value = dataset[()]

            print(
                f"{prefix}  Value: "
                f"{format_value(value)}"
            )

        else:
            print(
                f"{prefix}  Value: "
                f"[Not displayed - {total_elements} elements]"
            )

    except Exception as error:

        print(
            f"{prefix}  Could not read value: {error}"
        )

    print()


# ============================================================
# GROUP INSPECTION
# ============================================================

def inspect_group(group, indent=0):
    """
    Recursively inspect an HDF5 group.
    """

    prefix = " " * indent

    for name, obj in group.items():

        if isinstance(obj, h5py.Group):

            print(f"{prefix}GROUP: {name}")

            print_attributes(obj, indent + 2)

            print()

            inspect_group(
                obj,
                indent + 4
            )

        elif isinstance(obj, h5py.Dataset):

            inspect_dataset(
                name,
                obj,
                indent
            )

        else:

            print(
                f"{prefix}UNKNOWN OBJECT: "
                f"{name}"
            )


# ============================================================
# FILE INSPECTION
# ============================================================

def inspect_file(file_path):
    """
    Inspect one HDF5 file completely.
    """

    print()
    print("=" * 100)
    print("FILE:")
    print(file_path)
    print("=" * 100)

    try:

        with h5py.File(
            file_path,
            "r"
        ) as h5_file:

            # ------------------------------------------------
            # ROOT KEYS
            # ------------------------------------------------

            print("\nROOT KEYS:")

            root_keys = list(
                h5_file.keys()
            )

            print(root_keys)


            # ------------------------------------------------
            # ROOT ATTRIBUTES
            # ------------------------------------------------

            print(
                "\nROOT ATTRIBUTES:"
            )

            if len(
                h5_file.attrs
            ) == 0:

                print(
                    "No root attributes."
                )

            else:

                for key, value in (
                    h5_file.attrs.items()
                ):

                    print(
                        f"{key}: "
                        f"{format_value(value)}"
                    )


            # ------------------------------------------------
            # FULL STRUCTURE
            # ------------------------------------------------

            print()
            print("=" * 60)
            print("FULL HDF5 STRUCTURE")
            print("=" * 60)
            print()

            inspect_group(
                h5_file
            )


    except Exception as error:

        print()
        print(
            "ERROR READING FILE:"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("HDF5 METADATA INSPECTION")
    print("=" * 100)

    print(
        f"\nNumber of files to inspect: "
        f"{len(FILES)}"
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )

    print()


    # Open output file and redirect all print output there
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        # Store original print function
        import builtins

        original_print = (
            builtins.print
        )


        def file_print(
            *args,
            **kwargs
        ):
            """
            Print to both:
            1. Terminal
            2. metadata_output.txt
            """

            original_print(
                *args,
                **kwargs
            )

            original_print(
                *args,
                **kwargs,
                file=output
            )


        # Temporarily replace print
        builtins.print = (
            file_print
        )


        try:

            print()
            print(
                "#" * 100
            )

            print(
                "STARTING METADATA INSPECTION"
            )

            print(
                "#" * 100
            )


            for index, file_path in enumerate(
                FILES,
                start=1
            ):

                print()
                print(
                    "#" * 100
                )

                print(
                    f"FILE "
                    f"{index} "
                    f"OF "
                    f"{len(FILES)}"
                )

                print(
                    "#" * 100
                )


                if os.path.exists(
                    file_path
                ):

                    inspect_file(
                        file_path
                    )

                else:

                    print()
                    print(
                        "FILE NOT FOUND:"
                    )

                    print(
                        file_path
                    )

                    print()
                    print(
                        "CHECK THE PATH ABOVE."
                    )


            print()
            print(
                "#" * 100
            )

            print(
                "INSPECTION COMPLETE"
            )

            print(
                "#" * 100
            )

            print()
            print(
                f"Full output saved to:"
            )

            print(
                OUTPUT_FILE
            )


        finally:

            # Restore normal print
            builtins.print = (
                original_print
            )


    # Final confirmation after restoring print
    print()
    print("=" * 100)
    print(
        "DONE!"
    )
    print(
        f"Output saved to:"
    )
    print(
        OUTPUT_FILE
    )
    print("=" * 100)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()