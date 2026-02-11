import csv
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Any

def convert_asp_to_standard(input_dir: str, output_dir: str):
    """
    Converts ASP generator output (folder of CSVs) to Standard Format (facts.csv, targets.csv).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    facts_rows = []
    targets_rows = []

    # Regex to extract sample_id from filename (e.g., sample_00001.csv or 00001.csv)
    # Adjust regex based on actual file naming pattern from generator.py
    # generator.py uses: sample_name_pattern = "{:0" + str(len(str(conf.num_samples - 1))) + "d}"
    # So filenames are likely just "00000", "00001" (no extension? or implicit?)
    # Verify file extension. generator.py -> kg_writer.KgWriter.write(kg, conf.output_dir, base_name)
    # Usually adds .csv. Let's assume .csv.

    files = list(input_path.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Processing...")

    for file_path in files:
        filename = file_path.stem
        # Try to extract numeric ID, otherwise use filename
        match = re.search(r'(\d+)', filename)
        if match:
            sample_id = match.group(1)
        else:
            sample_id = filename

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Expecting headers: subject, predicate, object, label, fact_type, is_base_fact
            # Note: headers might vary if reldata changed.
            
            for row in reader:
                # Map to Standard Format
                # Standard: sample_id, subject, predicate, object, label, truth_value, type, hops, corruption_method
                
                is_base = row.get("is_base_fact", "0") == "1"
                label = row.get("label", "1")
                is_positive = label == "1"
                
                std_row = {
                    "sample_id": sample_id,
                    "subject": row.get("subject"),
                    "predicate": row.get("predicate"),
                    "object": row.get("object"),
                    "label": int(label),
                    "truth_value": "True" if is_positive else "False", # ASP usually closed world or explicit neg?
                    "type": "base_fact" if is_base else ("inf_root" if is_positive else "neg_inf_root"),
                    "hops": 0 if is_base else -1, # Unknown hops for ASP
                    "corruption_method": None # Unknown
                }

                # Add to targets (everything is a target/query)
                targets_rows.append(std_row)

                # Add to facts if positive base fact
                if is_base and is_positive:
                    facts_rows.append({
                        "sample_id": sample_id,
                        "subject": std_row["subject"],
                        "predicate": std_row["predicate"],
                        "object": std_row["object"]
                    })

    # Write outputs
    if facts_rows:
        with open(output_path / "facts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "subject", "predicate", "object"])
            writer.writeheader()
            writer.writerows(facts_rows)
            
    if targets_rows:
        keys = list(targets_rows[0].keys())
        with open(output_path / "targets.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(targets_rows)

    print(f"Conversion complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ASP generator output to Standard RRN Format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ASP sample CSVs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save facts.csv and targets.csv")
    args = parser.parse_args()

    convert_asp_to_standard(args.input_dir, args.output_dir)
