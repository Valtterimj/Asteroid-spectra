import numpy as np
import pandas as pd
import os
from modules._constants import _project_dir
from modules.NN_evaluate import evaluate
from modules.utilities_spectra import collect_all_models, normalise_spectra
from modules.collect_data import resave_ASPECT_transmission
from scipy.interpolate import interp1d


# 1. Load spectra file
csv_path = os.path.join(_project_dir, "test_data", "pixel_reflectances_4-pixel_binning_csv", "600w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv")
df = pd.read_csv(csv_path, sep=" ", header=None)

# 2. Extract wavelengths and spectra
wavelengths = df.iloc[:, 0].to_numpy()         # shape (N,)
spectra = df.iloc[:, 1:].to_numpy().T          # shape (16, N)

spectra_normalized = normalise_spectra(
    data=spectra,
    wavelength=wavelengths,
    wvl_norm_nm=1539.0
)

model_subdir = "composition/ASPECT-vis-nir1-nir2-1539"   # Composition
# model_subdir = "taxonomy/ASPECT-vis-nir1-nir2-1539"   # Taxonomy
# dat = os.path.join(_project_dir, "test_data", "test.npz")

model_name = ""
model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

# predictions = evaluate(model_names, spectra_normalized)

# print(predictions)
# formatted = np.round(predictions * 100, 1)

# # Print nicely (tab-separated, one row per spectrum if multiple)
# for row in formatted:
#     print("\t".join(f"{v:.1f}" for v in row))

# formatted = np.round(predictions * 100, 1)

# # Print nicely (tab-separated, one row per spectrum if multiple)
# for pred_row, fmt_row in zip(predictions, formatted):
#     pred_str = "\t".join(f"{v:.4f}" for v in pred_row)   # original values with 4 decimals
#     fmt_str = "\t".join(f"{v:.1f}" for v in fmt_row)     # rounded percentage values
#     print(f"RAW: {pred_str}")
#     print(f"FMT: {fmt_str}")
#     print()

# # python3 level_3/modules/NN_test_evaluate.py



def evaluate_and_save_results(csv_path, model):
    composition_headers = [
        "OL (vol%)", "OPX (vol%)", "CPX (vol%)",
        "Fa", "Fo",
        "Fs (OPX)", "En (OPX)",
        "Fs (CPX)", "En (CPX)", "Wo (CPX)"
    ]

    taxonomy_headers = [
        "A+ = A + Sa",
        "C+ = C + Cb + Cg + B",
        "Ch+ = Ch + Cgh",
        "D",
        "L",
        "Q",
        "S+ = S + Sqw + Sr + Srw + Sw",
        "V+ = V + Vw",
        "X+ = X + Xc + Xe + Xk"
    ]

    df = pd.read_csv(csv_path, sep=" ", header=None)

    wavelengths = df.iloc[:, 0].to_numpy()         # shape (N,)
    spectra = df.iloc[:, 1:].to_numpy().T          # shape (16, N)

    spectra_normalized = normalise_spectra(
        data=spectra,
        wavelength=wavelengths,
        wvl_norm_nm=1539.0
    )

    # Original file name
    filename = os.path.basename(csv_path)

    # Split name and extension
    name, ext = os.path.splitext(filename)

    if model == 'composition':
        model_subdir = "composition/ASPECT-vis-nir1-nir2-1539"   # Composition
        new_filename = f"{name}_git_composition{ext}"
    else:
        model_subdir = "taxonomy/ASPECT-vis-nir1-nir2-1539"   # Taxonomy
        new_filename = f"{name}_git_taxonomy{ext}"
    
    model_name = ""
    model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=True)

    predictions = evaluate(model_names, spectra_normalized)

    formatted = np.round(predictions * 100, 1)

    output_path = os.path.join(_project_dir, "pixel_reflectances_4-pixel_binning_csv", new_filename)
    # Create DataFrame with index starting at 1
    if model == 'composition':
        df = pd.DataFrame(formatted, columns=composition_headers)
    else:
        df = pd.DataFrame(formatted, columns=taxonomy_headers)
    df.index += 1  # Start index from 1
    df.index.name = "# No."

    # Save to CSV with space separator
    df.to_csv(output_path, sep="\t", float_format="%.1f")

csv_path = os.path.join(_project_dir, "pixel_reflectances_4-pixel_binning_csv", "200w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv")
# evaluate_and_save_results(csv_path, 'composition')
# evaluate_and_save_results(csv_path, 'taxonomy')




def calculate_mean_differences(base_filenames, folder_path):
    results = []

    for base_name in base_filenames:
        base = os.path.splitext(base_name)[0]  # remove .csv extension

        comp_git_path = os.path.join(folder_path, f"{base}_git_composition.csv")
        comp_web_path = os.path.join(folder_path, f"{base}_web_composition.csv")
        tax_git_path  = os.path.join(folder_path, f"{base}_git_taxonomy.csv")
        tax_web_path  = os.path.join(folder_path, f"{base}_web_taxonomy.csv")

        try:
            comp_git = pd.read_csv(comp_git_path, sep="\t").iloc[:, 1:].to_numpy()
            comp_web = pd.read_csv(comp_web_path, sep="\t").iloc[:, 1:].to_numpy()
            tax_git = pd.read_csv(tax_git_path, sep="\t").iloc[:, 1:].to_numpy()
            tax_web = pd.read_csv(tax_web_path, sep="\t").iloc[:, 1:].to_numpy()

            # Compute absolute mean differences
            comp_diff = np.mean(np.abs(comp_git - comp_web))
            tax_diff  = np.mean(np.abs(tax_git  - tax_web))

            results.append((base_name, comp_diff, tax_diff))
        
        except Exception as e:
            print(f'Error processing {base_name}: {e}')
            results.append((base_name), None, None)

    return results


folder_name = os.path.join(_project_dir, "pixel_reflectances_4-pixel_binning_csv")
base_filenames = [
    '200w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv',
    '400w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv',
    '600w_exposures_1250-5000-5000_pixel_reflectances(4-pixel_binning).csv',
    '600w_exposures_1875-7500-7500_pixel_reflectances(4-pixel_binning).csv',
    '600w_exposures_2500-10000-10000_pixel_reflectances(4-pixel_binning).csv'
]

# diffs = calculate_mean_differences(base_filenames, folder_name)
# for name, comp_diff, tax_diff in diffs:
#     print(f"{name} → Composition Δ: {comp_diff:.3f}, Taxonomy Δ: {tax_diff:.3f}")




output_csv_path = os.path.join(_project_dir, "test_data/test.npz")

def csv_to_npz_raw(csv_path, output_npz_path):
    """
    Converts a CSV file to a .npz file containing only the spectrum data (no wavelengths).
    
    Parameters:
        csv_path (str): Path to the input CSV file.
        output_npz_path (str): Path to save the output .npz file.
    """
    df = pd.read_csv(csv_path, sep=" ", header=None)

    # Extract wavelength column and spectra
    wavelengths = df.iloc[:, 0].to_numpy()            # shape (N_points,)
    spectra = df.iloc[:, 1:].to_numpy().T             # shape (N_spectra, N_points)

    # Save to .npz with named keys
    np.savez(output_npz_path, spectra=spectra, wavelengths=wavelengths)

# csv_to_npz_raw(csv_path, output_csv_path)

# dat = os.path.join(_project_dir, "test_data", "test.npz")
# data = np.load(dat)
# spectrum = data["spectra"] 
# print(spectrum)
# print()
# print(data)

def load_npz(filepath: str) -> dict:
    """
    Load data from a .npz file and return it as a dictionary.
    
    Parameters:
        filepath (str): Path to the .npz file.
    
    Returns:
        dict: A dictionary where keys are variable names and values are NumPy arrays.
    """
    try:
        with np.load(filepath, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}
    
def read_npz():
    data_dict = load_npz(os.path.join(_project_dir, 'datasets/ASPECT/ASPECT_transmission.npz'))

    # List all arrays in the file
    print("Keys:", list(data_dict.keys()))

    if "wavelengths" in data_dict:
        wl = data_dict["wavelengths"]
        print(wl)
    if "vis" in data_dict:
        vis = data_dict["vis"]
        print(vis)
    if "nir1" in data_dict:
        nir1 = data_dict["nir2"]
        print(nir1)
    if "nir2" in data_dict:
        nir2 = data_dict["nir2"]
        print(nir2)
    if "swir" in data_dict:
        swir = data_dict["swir"]
        print(swir)

def create_aspect_transmission():
    resave_ASPECT_transmission()

create_aspect_transmission()

def model_spects():
    from modules.utilities_spectra import load_keras_model, _path_model
    from os import path
    model_filename = path.join(_path_model, "composition", "ASPECT-vis-nir1-nir2-1539-NEW",
                            "CNN_ASPECT-vis-nir1-nir2-1539_1110-11-110-111-000_20250616085829.h5")
    model = load_keras_model(model_filename)  # to read the model
    import ast
    import h5py
    # to read the metadata
    with h5py.File(model_filename, "r") as f:
        parameters = ast.literal_eval(f.attrs["params"])  # to convert string "{key: value}" to dict {key: value}
        layer_names = f.attrs["layer_names"]  # names of layers; now only general info visible also using model.summary(). I use this for the ongoing project because I only save weights, not the whole model to keep it more general; you can ignore the layer names here
    wavelengths = parameters["wavelengths"] #and many others are there (insttument, normalisation wavelength, hyperparameters)
    print(f'wavelengths: {wavelengths}')
    print(f'len of wavelengths: {len(wavelengths)}')

# model_spects()