import os
import sys

str_name=sys.argv[1]
file_path = f"data.save/SURROUNDING_ATOMS_of_{str_name}.txt"
# Read the file
with open(file_path, "r") as file:
    lines = file.readlines()

# Extract central atom element information and create a unique element set
central_elements = [line.split("-")[1].split("#")[0] for line in lines[1:]]
unique_elements = set(central_elements)

results_list_updated = []
header_list = ["#中心原子环境类型"] + [f"最近邻{element}环境类型" for element in unique_elements]

for line in lines[1:]:
    line_data = line.split("\t")
    center_atom = line_data[0]
    surrounding_atoms = line_data[1:4]

    # Create a dictionary to store lists of each element
    atoms_dict = {element: [] for element in unique_elements}

    # For O atoms, only consider once
    if "O" in center_atom:
        surrounding_atoms = [surrounding_atoms[0]]

    for atoms in surrounding_atoms:
        #print("1",atoms,type(atoms))
        atoms=atoms.replace(";",",")
        #print("2",atoms,type(atoms))
        for atom in eval(atoms):  # Convert string representation of list to actual list
            element = atom.split("-")[1].split("#")[0]
            atoms_dict[element].append(atom)

    # Replace empty lists with "No Match"
    row_data = [center_atom] + [(str(atoms_dict[element]).replace(", ","+ ") if atoms_dict[element] else "NO_DATA") for element in unique_elements]
    results_list_updated.append(row_data)

# Write the updated data into a new file
output_filename = f"data.save/Separate_Surrouding_List_of_{str_name}.txt"

with open(output_filename, "w") as outfile:
    outfile.write("\t".join(header_list) + "\n")
    for row in results_list_updated:
        outfile.write("\t".join(map(str, row)) + "\n")
print(output_filename,"created!")
