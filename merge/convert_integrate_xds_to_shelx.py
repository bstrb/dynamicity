import os
import argparse
import pandas as pd
import numpy as np

def convert_integrate_xds_to_shelx(input_file, output_file=None, sort_by="d_spacing"):
    if output_file is None:
        output_file = os.path.join(os.path.dirname(input_file), "shelx", "shelx.hkl")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Reading {input_file}...")
    headers = ["H", "K", "L", "I", "SIGMA", "X", "Y", "Z", "RLP", "PEAK", "CORR", "MAXC", "TRAN", "IP", "SIGP", "I_CORR", "SIG_CORR"]
    
    # Read the data, skipping the header lines which start with !
    data = []
    unit_cell = None
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('!UNIT_CELL_CONSTANTS='):
                unit_cell = [float(x) for x in line.split('=')[1].split()]
            if line.startswith('!'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                # H, K, L, I, SIGMA
                data.append([int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])])

    df = pd.DataFrame(data, columns=["H", "K", "L", "I", "SIGMA"])
    
    if sort_by == "d_spacing" and unit_cell is not None:
        print("Sorting by d-spacing (ascending)...")
        a, b, c, alpha, beta, gamma = unit_cell
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
        
        # Calculate metric tensor
        G = np.array([
            [a**2, a*b*np.cos(gamma_rad), a*c*np.cos(beta_rad)],
            [a*b*np.cos(gamma_rad), b**2, b*c*np.cos(alpha_rad)],
            [a*c*np.cos(beta_rad), b*c*np.cos(alpha_rad), c**2]
        ])
        G_inv = np.linalg.inv(G)
        
        hkl = df[["H", "K", "L"]].values
        # 1/d^2 = h.T * G_inv * h
        inv_d2 = np.sum((hkl @ G_inv) * hkl, axis=1)
        df["inv_d2"] = inv_d2
        df = df.sort_values("inv_d2", ascending=True).drop(columns="inv_d2")
    
    print(f"Writing {output_file}...")
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{int(row['H']):4d}{int(row['K']):4d}{int(row['L']):4d}{row['I']:8.2f}{row['SIGMA']:8.2f}\n")
        f.write("   0   0   0    0.00    0.00\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XDS INTEGRATE.HKL to SHELX HKLF4 format")
    parser.add_argument("input_file", help="Path to the INTEGRATE.HKL file")
    parser.add_argument("-o", "--output", dest="output_file", default=None, help="Path to output SHELX HKL file")
    parser.add_argument("--sort", dest="sort_by", choices=["d_spacing", "original"], default="d_spacing", help="Sorting method")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        exit(1)
    
    convert_integrate_xds_to_shelx(args.input_file, args.output_file, args.sort_by)
