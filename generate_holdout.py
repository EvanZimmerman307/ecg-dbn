import os
import shutil

def remove_labels_from_header(hea_file_path):
    """
    Removes lines in the header file that contain label information.
    In the `.hea` file, labels are often included in comments starting with '#'.
    This function removes such lines.
    """
    with open(hea_file_path, 'r') as f:
        lines = f.readlines()

    # Create a new list of lines without label-related comments
    new_lines = []
    for line in lines:
        if line.startswith("# Chagas"): #or line.startswith("#Label"):  # You can add more label-related comments here if needed
            continue  # Skip lines with labels
        new_lines.append(line)

    return new_lines

def create_holdout_data(input_folder, output_folder):
    """
    Creates the 'holdout_data' folder by copying .dat and .hea files from the input folder.
    Removes labels from the .hea files before copying them.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".dat"):
            # For each .dat file, copy it to the output folder
            dat_file = os.path.join(input_folder, filename)
            shutil.copy(dat_file, output_folder)

            # Now handle the corresponding .hea file
            hea_file = os.path.splitext(dat_file)[0] + ".hea"
            if os.path.exists(hea_file):
                # Remove labels from the .hea file
                new_hea_lines = remove_labels_from_header(hea_file)

                # Create a new .hea file with the modified content in the output folder
                new_hea_file = os.path.join(output_folder, os.path.basename(hea_file))
                with open(new_hea_file, 'w') as f:
                    f.writelines(new_hea_lines)

    print(f"Holdout data has been created in {output_folder}")

if __name__ == "__main__":
    input_folder = "ptbxl_output/01000"  # Replace with the path to your training data folder
    output_folder = "holdout_data"  # Folder where you want to create the holdout data

    create_holdout_data(input_folder, output_folder)
