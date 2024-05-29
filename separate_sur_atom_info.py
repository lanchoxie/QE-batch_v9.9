import os
import sys

str_name=sys.argv[1]
file_path = f"data.save/SURROUNDING_ATOMS_of_{str_name}.txt"
# Step 1: Read the file
with open(file_path, "r") as file:
    lines = file.readlines()

# Step 2: Extract central atom element information and create a unique element set
# Extract all central atom elements
central_elements = [line.split("-")[1].split("#")[0] for line in lines[1:]]
# Create a unique element set
unique_elements = set(central_elements)

# Step 3: For each line, count the number of each element
results = []
#header = ["#Center_Atom_env"] + [f"Surround_{x}_number" for x in list(unique_elements)]
header = ["#中心原子环境类型"] + [f"最近邻{element}数量" for element in unique_elements]

for line in lines[1:]:
    line_data = line.split("\t")
    center_atom = line_data[0]
    surrounding_atoms = line_data[1:4]

    # Initialize a dictionary to store counts of each element
    count_dict = {element: 0 for element in unique_elements}

    # For O atoms, only count once
    if "O" in center_atom:
        surrounding_atoms = [surrounding_atoms[0]]

    for atoms in surrounding_atoms:
        atoms=atoms.replace("; ",", ")
        for atom in eval(atoms):  # Convert string representation of list to actual list
            element = atom.split("-")[1].split("#")[0]
            count_dict[element] += 1

    row_data = [center_atom] + [count_dict[element] for element in unique_elements]
    results.append(row_data)

# Step 4: Write the data into a new file
output_filename = f"data.save/Separate_Surrouding_of_{str_name}.txt"

with open(output_filename, "w") as outfile:
    outfile.write("\t".join(header) + "\n")
    for row in results:
        outfile.write("\t".join(map(str, row)) + "\n")
print(output_filename,"created!")
