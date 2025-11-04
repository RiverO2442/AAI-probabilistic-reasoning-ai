#############################################################################
# auto_fix_bn_config.py
#
# Automatically detects and fixes missing "P(var)" entries in a Bayesian
# network configuration file. Ensures that every variable has a definition.
#############################################################################
import re
from pathlib import Path

def auto_fix_bn(config_path):
    print(f"\n🔧 Checking BN config: {config_path}")
    content = Path(config_path).read_text()

    rv_match = re.search(r"random_variables:(.*)", content)
    structure_match = re.search(r"structure:(.*)", content, re.DOTALL)
    if not rv_match or not structure_match:
        print("❌ Missing random_variables or structure section.")
        return

    random_vars = [v.strip() for v in rv_match.group(1).replace(";", ",").split(",") if v.strip()]
    structure_lines = [s.strip() for s in structure_match.group(1).split(";") if s.strip()]

    defined = set()
    for line in structure_lines:
        m = re.match(r"P\(([^|)]+)", line)
        if m:
            defined.add(m.group(1).strip())

    missing = [v for v in random_vars if v not in defined]
    if not missing:
        print("✅ No missing nodes found.")
        return

    print(f"⚠️ Adding missing root nodes: {missing}")
    for v in missing:
        structure_lines.append(f"P({v})")

    new_structure = "structure:\n" + ";\n".join(structure_lines)
    fixed_content = re.sub(r"structure:(.*)", new_structure, content, flags=re.DOTALL)

    new_path = Path(config_path).with_name(Path(config_path).stem + "_fixed.txt")
    Path(new_path).write_text(fixed_content)
    print(f"✅ Saved fixed config as: {new_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python auto_fix_bn_config.py <config_file.txt>")
    else:
        auto_fix_bn(sys.argv[1])
