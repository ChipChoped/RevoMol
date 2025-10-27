import json
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt


def extract_data_from_file(json_file_path):
    """
    Checks for file existence and validity before loading JSON data.
    """
    data = None
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}. Skipping.")
        return None, None, None, 0, None
    if not os.path.isfile(json_file_path):
        print(f"Error: Path {json_file_path} is not a file. Skipping.")
        return None, None, None, 0, None

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {json_file_path}. Skipping.")
        return None, None, None, 0, None
    except Exception as e:
        print(
            f"An unexpected error occurred while reading file {json_file_path}: {e}. Skipping."
        )
        return None, None, None, 0, None
    return extract_statistics_from_data(data)


def extract_statistics_from_data(data, to_value=299):
    """
    Loads molecule data from a single JSON data object and extracts relevant metrics.
    """
    if data is None:
        return None, None, None, 0, None

    inaccuracies = []
    times = []
    solution_diffs = []
    error_count = 0
    nb_to = 0

    for molecule_key, molecule_data in data.items():
        if not isinstance(molecule_data, dict):
            print(f"Warning: Skipping entry '{molecule_key}' due to unexpected format.")
            continue

        required_keys = [
            "inacuracy",
            "time",
            "nb_raw_solutions",
            "nb_molecules",
            "error",
        ]
        if not all(key in molecule_data for key in required_keys):
            print(f"Warning: Skipping entry '{molecule_key}' due to missing keys.")
            continue

        try:
            has_error = molecule_data.get("error", False)
            if has_error:
                error_count += 1
                if molecule_data["time"] >= to_value:
                    nb_to += 1
                continue

            inaccuracies.append(float(molecule_data["inacuracy"]))
            times.append(float(molecule_data["time"]))
            nb_raw_solutions = int(molecule_data["nb_raw_solutions"])
            nb_molecules = int(molecule_data["nb_molecules"])
            solution_diffs.append(nb_raw_solutions - nb_molecules)

        except (ValueError, TypeError) as e:
            print(
                f"Warning: Skipping entry '{molecule_key}' due to data type error: {e}"
            )
            continue

    return inaccuracies, times, solution_diffs, error_count, nb_to


def get_stats(data_list):
    """Calculates statistics for a given list of numbers."""
    if not data_list:
        return None
    try:
        np_array = np.array(data_list)
        stats = {
            "mean": np.mean(np_array),
            "median": np.median(np_array),
            "q1": np.percentile(np_array, 25),
            "q3": np.percentile(np_array, 75),
        }
        return stats
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return None


def print_summary(stats, total_errors, total_timeout, total_valid_entries, label):
    """Prints the calculated statistics in a readable format for a single file."""
    if not stats:
        print("No statistics were calculated (no valid data found).")
        return

    print(f"Summary for: {label}")
    print(f"  Processed {total_valid_entries} valid entries.")
    print(f"  Skipped {total_errors} entries ({total_timeout} timeouts).")

    for metric, values in stats.items():
        if values:
            print(f"  - Stats for '{metric}':")
            print(f"    Mean: {values['mean']:.4f}, Median: {values['median']:.4f}")
        else:
            print(f"  - Stats for '{metric}': Not available (no valid data).")
    print("----------------------------------")


def plot_combined_violins(all_data, name_mapping, output_dir="comparison_plots"):
    """
    Generates and saves combined violin plots for each metric across all files.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_scale_metrics = ["time_second", "inaccuracy", "solution_difference"]

    for metric, file_data in all_data.items():
        labels = []
        datasets = []
        # Ensure the order of datasets matches the order of the name_mapping
        for file_path in name_mapping.keys():
            # Check if data exists and is not empty
            if file_path in file_data and file_data[file_path]:
                labels.append(name_mapping[file_path])
                datasets.append(file_data[file_path])

        if not datasets:
            print(
                f"Skipping plot for '{metric}': No valid data available from any file."
            )
            continue

        plt.figure(figsize=(12, 8))

        parts = plt.violinplot(
            datasets, showmeans=True, showmedians=True, showextrema=True
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("#D4E6F1")
            pc.set_edgecolor("black")
            pc.set_alpha(1)

        parts["cmedians"].set_edgecolor("red")
        parts["cmedians"].set_linewidth(2)
        parts["cmeans"].set_edgecolor("blue")
        parts["cmeans"].set_linewidth(2)
        parts["cbars"].set_edgecolor("black")

        plt.title(
            f"Comparison of {metric.replace('_', ' ').title()} Across Methods",
            fontsize=16,
        )
        plt.xlabel("Method")

        plt.xticks(
            ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha="right"
        )

        plt.ylabel(metric.replace("_", " ").title())
        if metric in log_scale_metrics:
            # Since negative values are filtered, we only need to check for zeros
            if any(val <= 0 for ds in datasets for val in ds):
                plt.yscale("symlog", linthresh=1e-1)
            else:
                plt.yscale("log")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        plot_filename = f"{output_dir}/{metric}_comparison_violin.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved combined plot to {plot_filename}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze JSON results and generate violin plots."
    )
    parser.add_argument("json_files", nargs="+", help="List of JSON files to process.")
    parser.add_argument(
        "--names", nargs="+", help="Space-separated list of custom names for the plots."
    )

    args = parser.parse_args()

    json_files = args.json_files
    custom_names = args.names if args.names else []

    name_mapping = {}
    if custom_names:
        if len(custom_names) == len(json_files):
            name_mapping = dict(zip(json_files, custom_names))
        else:
            print(
                f"Warning: The number of names ({len(custom_names)}) does not match the number of files ({len(json_files)})."
            )
            print("Using default filenames as labels.")
            name_mapping = {
                file: os.path.basename(file).replace(".json", "") for file in json_files
            }
    else:
        name_mapping = {
            file: os.path.basename(file).replace(".json", "") for file in json_files
        }

    all_results = {
        "inaccuracy": {},
        "time_second": {},
        "solution_difference": {},
    }

    for file_path in json_files:
        print(f"\nProcessing file: {file_path}")
        inaccuracies, times, diffs, errors_in_file, nb_to = extract_data_from_file(
            file_path
        )

        if inaccuracies is not None:
            # --- FILTERING LOGIC ---
            if diffs is not None:
                original_count = len(diffs)
                # Keep only values greater than or equal to zero
                diffs = [d for d in diffs if d >= 0]
                if len(diffs) < original_count:
                    omitted_count = original_count - len(diffs)
                    print(
                        f"  Info: Omitted {omitted_count} negative value(s) from 'solution_difference'."
                    )

            # Store the (potentially filtered) data
            all_results["inaccuracy"][file_path] = inaccuracies
            all_results["time_second"][file_path] = times
            all_results["solution_difference"][file_path] = diffs

            # Calculate stats on the filtered data
            current_stats = {
                "inaccuracy": get_stats(inaccuracies),
                "time_second": get_stats(times),
                "solution_difference": get_stats(diffs),
            }
            label_name = name_mapping.get(file_path, os.path.basename(file_path))
            print_summary(
                current_stats, errors_in_file, nb_to, len(inaccuracies), label_name
            )
        else:
            print("Could not process this file.")

    print("\n--- Generating Combined Plots ---")
    plot_combined_violins(all_results, name_mapping)
    print("\nAnalysis complete.")
