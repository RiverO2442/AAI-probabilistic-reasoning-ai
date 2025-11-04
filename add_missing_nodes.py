#############################################################################
# add_missing_nodes.py
#
# Automatically adds missing P(variable) definitions to a Bayesian Network
# config file, fixing errors like "Couldn't find parent(s) of variable X".
#
# Usage:
#   python add_missing_nodes.py ./config/config_fraud.txt
#############################################################################

from pathlib import Path
import re

def fix_missing_nodes(config_path):
    print(f"\n🔧 Checking and fixing structure in {config_path}...")
    content = Path(config_path).read_text()

    # Extract random_variables and structure
    rv_match = re.search(r"random_variables:(.*)", content)
    struct_match = re.search(r"structure:(.*)", content, re.DOTALL)
    if not rv_match or not struct_match:
        print("❌ Missing 'random_variables' or 'structure' section.")
        return

    random_vars = [v.strip() for v in rv_match.group(1).replace(";", ",").split(",") if v.strip()]
    structure = [s.strip() for s in struct_match.group(1).split(";") if s.strip()]

    # Get already-defined variables
    defined_vars = []
    for s in structure:
        m = re.match(r"P\(([^|)]+)", s)
        if m:
            defined_vars.append(m.group(1).strip())

    missing = [v for v in random_vars if v not in defined_vars]
    if not missing:
        print("✅ All random variables are defined in the structure.")
        return

    print(f"⚠️ Missing variables detected: {missing}")
    for v in missing:
        structure.append(f"P({v})")

    # Rewrite structure section
    new_structure = "structure:" + ";".join(structure)
    fixed_content = re.sub(r"structure:(.*)", new_structure, content, flags=re.DOTALL)

    new_path = Path(config_path).with_name(Path(config_path).stem + "_fixed.txt")
    Path(new_path).write_text(fixed_content)

    print(f"✅ Added missing nodes. Saved fixed config as: {new_path}")
    print("🆕 Added independent definitions for:", missing)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python add_missing_nodes.py [config_file.txt]")
        sys.exit(1)
    fix_missing_nodes(sys.argv[1])
