GANDALF_RUNNER_CODE = r"""
import sys, os, argparse, importlib.util

def load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("host_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # CRITICAL: register in sys.modules before executing so dataclasses & typing can resolve module
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--outbase", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--step", type=float, required=True)
    # Forward everything else verbatim to the iterator
    args, extra = ap.parse_known_args()

    # Import the host module (your GUI file) safely
    mod = load_module_from_path(args.host)

    # Retrieve vendored iterator
    gi = getattr(mod, "gandalf_iterator", None)
    if gi is None:
        print("ERROR: gandalf_iterator not found in host module", file=sys.stderr)
        sys.exit(2)

    # Call it
    rc = gi(
        geomfile_path=args.geom,
        cellfile_path=args.cell,
        input_path=args.input,
        output_file_base=args.outbase,
        num_threads=args.threads,
        max_radius=args.radius,
        step=args.step,
        extra_flags=extra,
    )

    try:
        code = int(rc) if rc is not None else 0
    except Exception:
        code = 0
    sys.exit(code)

if __name__ == "__main__":
    main()
"""