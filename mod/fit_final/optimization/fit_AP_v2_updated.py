
import os
import json
import argparse
import numpy as np
from collections import namedtuple

# === Load your NEURON model
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

def parse_passive_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def extract_metadata_from_filename(filename):
    base = os.path.basename(filename)
    age = 0
    phenotype = "WT"
    for part in base.split("_"):
        if part.startswith("P") and part[1:].isdigit():
            age = int(part[1:])
        elif "TeNT" in part:
            phenotype = "TeNT"
        elif "iMNTB" in part:
            phenotype = "iMNTB"
    return age, phenotype

def get_active_bounds(age, phenotype):
    if age <= 3:
        bounds = {
            "gna": (20, 100),
            "gkht": (10, 80),
            "gka": (2, 20)
        }
    elif age <= 6:
        bounds = {
            "gna": (40, 120),
            "gkht": (20, 100),
            "gka": (5, 30)
        }
    else:
        bounds = {
            "gna": (60, 160),
            "gkht": (30, 120),
            "gka": (5, 40)
        }

    if phenotype == "TeNT":
        bounds["gka"] = (2, 20)
    elif phenotype == "iMNTB":
        bounds["gna"] = (bounds["gna"][0], bounds["gna"][1] * 0.8)

    return bounds

def main(data_file, passive_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Fitting AP from: {data_file}")
    print(f"📥 Using passive params from: {passive_json}")

    passive_params = parse_passive_json(passive_json)
    age, phenotype = extract_metadata_from_filename(data_file)
    bounds = get_active_bounds(age, phenotype)

    print(f"📌 Age: P{age}, Phenotype: {phenotype}")
    print(f"🔧 Active bounds: {bounds}")

    # TODO: Load your AP data, run the simulation with NEURON, do the optimization here...
    # Placeholder: You would instantiate the model, inject current, optimize to fit spike shape

    # Final output (placeholder)
    print("✅ Fitting complete (mock). Save parameters and plots here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to experimental AP CSV")
    parser.add_argument("--passive_json", required=True, help="Path to passive param JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to store results")
    args = parser.parse_args()
    main(args.data, args.passive_json, args.output_dir)
