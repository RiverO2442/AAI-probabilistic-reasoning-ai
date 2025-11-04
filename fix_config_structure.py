#############################################################################
# auto_fix_structure.py
#
# Automatically repairs missing variable definitions in a Bayesian Network
# config file by adding lines like P(Location) for missing variables.
#
# Usage:
#   python auto_fix_structure.py ./config/config_fraud.txt
#
# Output:
#   A new file ./config/config_fraud_fixed.txt with a corrected structure.
#############################################################################

import re
import sys
from pathlib import Path

def fix_config_structure(config_path):
    print(f"\n🔧 Fixing structure in {config_path} ...")
    text = Path(config_path).read_text()

    # Extract random variables
    rv_match = re.search(r"random_variables:(.*)", text)
    if not rv_match:
        print("❌ ERROR: No 'random_variables:' found in file.")
        return

    random_vars = [v.strip() for v in rv_match.group(1).replace(";", ",").split(",") if v.strip()]

    # Extract structure block
    structure_match = re.search(r"structure:(.*)", text, re.DOTALL)
    if not structure_match:
        print("❌ ERROR: No 'structure:' section found.")
        return

    structure_raw = structure_match.group(1)
    structure_items = [s.strip() for s in structure_raw.split(";") if s.strip()]

    # Extract already defined variables
    defined_vars = []
    for s in structure_items:
        match = re.match(r"P\(([^|)]+)", s)
        if match:
            defined_vars.append(match.group(1).strip())

    # Find missing
    missing = [v for v in random_vars if v not in defined_vars]
    if not missing:
        print("✅ All variables are already defined.")
        return

    print(f"⚠️ Missing variable definitions found: {missing}")
    for var in missing:
        structure_items.append(f"P({var})")

    # Rebuild file
    new_structure = "structure:" + ";".join(structure_items)
    fixed_text = re.sub(r"structure:(.*)", new_structure, text, flags=re.DOTALL)

    fixed_path = Path(config_path).with_name(Path(config_path).stem + "_fixed.txt")
    Path(fixed_path).write_text(fixed_text)

    print(f"✅ Fixed config saved as {fixed_path}")
    print(f"🆕 Added independent variables: {missing}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python auto_fix_structure.py [config_file.txt]")
        sys.exit(1)
    fix_config_structure(sys.argv[1])
