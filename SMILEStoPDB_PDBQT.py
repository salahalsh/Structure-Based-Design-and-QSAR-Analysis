import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

# Read the input file
with open("smiles.txt", "r") as file:
    smiles_list = file.readlines()

# Create directories if they don't exist
os.makedirs("PDB", exist_ok=True)
os.makedirs("PDBQT", exist_ok=True)

for i, smiles in enumerate(smiles_list, start=1):
    try:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles.strip())
        
        if mol is not None:
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
            
            # Save as PDB file with the original SMILES string as a comment
            pdb_file = f"PDB/LIG{i}.pdb"
            with open(pdb_file, "w") as pdb_out:
                pdb_out.write(f"REMARK Original SMILES: {smiles.strip()}\n")
                pdb_out.write(Chem.MolToPDBBlock(mol))
            print(f"Generated PDB file: {pdb_file}")
            
            # Convert to PDBQT format with the original SMILES string as a comment
            obabel_input = pdb_file
            obabel_output = f"PDBQT/LIG{i}.pdbqt"
            obabel_command = f"obabel {obabel_input} -O {obabel_output} -xh -xn -xc 'REMARK Original SMILES: {smiles.strip()}'"
            os.system(obabel_command)
            print(f"Generated PDBQT file: {obabel_output}")
        else:
            print(f"Failed to generate molecule for SMILES: {smiles.strip()}")
            with open(f"LIG{i}_error.txt", "w") as error_file:
                error_file.write(smiles.strip())
    except Exception as e:
        print(f"Error processing SMILES: {smiles.strip()}")
        print(f"Error message: {str(e)}")
        with open(f"LIG{i}_error.txt", "w") as error_file:
            error_file.write(smiles.strip())