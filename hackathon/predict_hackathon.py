# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------
def _read_lines(p: Path) -> list[str]:
    if not p.exists():
        return []
    return p.read_text(errors="ignore").splitlines()

def _write_lines(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines))
    
def _diverse_subsample(lines: list[str], target: int) -> list[str]:
    if not lines or len(lines) <= target:
        return lines
    is_fasta = any(l.startswith(">") for l in lines)
    if is_fasta:
        pairs, cur_h, cur_seq = [], None, []
        for l in lines:
            if l.startswith(">"):
                if cur_h is not None:
                    pairs.append((cur_h, "".join(cur_seq)))
                cur_h, cur_seq = l, []
            else:
                cur_seq.append(l.strip())
        if cur_h is not None:
            pairs.append((cur_h, "".join(cur_seq)))
        if len(pairs) <= target:
            out = []
            for h, s in pairs:
                out.extend([h, s])
            return out
        stride = max(1, len(pairs) // target)
        sampled = pairs[::stride][:target]
        out = []
        for h, s in sampled:
            out.extend([h, s])
        return out
    else:
        stride = max(1, len(lines) // target)
        return lines[::stride][:target]

def _make_subsample(msa_rel: str | None, root: Optional[Path], tag: str, target: int) -> str | None:
    if msa_rel is None:
        return None
    src = Path(msa_rel)
    if not src.is_absolute():
        base = (root or Path(".")).resolve()
        src = (base / msa_rel).resolve()
    if not src.exists():
        return msa_rel
    sub = _diverse_subsample(_read_lines(src), target)
    out = src.with_name(src.stem + f".{tag}.msa")
    _write_lines(out, sub)
    try:
        return os.path.relpath(out, Path.cwd())
    except Exception:
        return str(out)

def _get_models_in_dir(pred_dir: Path, datapoint_id: str) -> list[Path]:
    return sorted(pred_dir.glob(f"{datapoint_id}_config_*_model_*.pdb"))

def _load_structure(pdb_path: Path):
    return PDBParser(QUIET=True).get_structure("m", str(pdb_path))

def _heavy_atoms(structure):
    for a in structure.get_atoms():
        el = (a.element or "").upper()
        if el and el != "H":
            yield a

def _atoms_of_chains(structure, ids: set[str]):
    for ch in structure.get_chains():
        if ch.id in ids:
            for a in ch.get_atoms():
                el = (a.element or "").upper()
                if el and el != "H":
                    yield a

def _interface_score_complex(structure, ab_ids={"H","L"}, ag_ids={"A"}):
    ab_atoms = list(_atoms_of_chains(structure, set(ab_ids)))
    ag_atoms = list(_atoms_of_chains(structure, set(ag_ids)))
    if not ab_atoms or not ag_atoms:
        return (0.0, 0.0, 999)
    ns = NeighborSearch(list(_heavy_atoms(structure)))
    contacts, clashes = 0, 0
    contact_cut, clash_cut = 4.5, 2.0
    ag_set = set(ag_atoms)
    for a in ab_atoms:
        for b in ns.search(a.coord, contact_cut):
            if b is a or b not in ag_set:
                continue
            d = calc_distance(a.get_vector(), b.get_vector())
            clashes += int(d < clash_cut)
            contacts += int(clash_cut <= d <= contact_cut)
    # cheap buried SASA proxy (avoid expensive copies)
    buried_sasa = 0.5 * contacts
    return float(contacts), float(buried_sasa), int(clashes)

def _interface_score_ligand(structure):
    protein_atoms, ligand_atoms = [], []
    for res in structure.get_residues():
        het = res.id[0].strip()
        if het == "":
            for a in res.get_atoms():
                el = (a.element or "").upper()
                if el and el != "H":
                    protein_atoms.append(a)
        else:
            if res.get_resname().strip() not in {"HOH","WAT"}:
                for a in res.get_atoms():
                    el = (a.element or "").upper()
                    if el and el != "H":
                        ligand_atoms.append(a)
    if not protein_atoms or not ligand_atoms:
        return (0.0, 0.0, 999)
    ns = NeighborSearch(list(_heavy_atoms(structure)))
    contacts, clashes = 0, 0
    contact_cut, clash_cut = 4.0, 2.0
    lig_set = set(ligand_atoms)
    for a in protein_atoms:
        for b in ns.search(a.coord, contact_cut):
            if b is a or b not in lig_set:
                continue
            d = calc_distance(a.get_vector(), b.get_vector())
            clashes += int(d < clash_cut)
            contacts += int(clash_cut <= d <= contact_cut)
    buried_sasa = 0.5 * contacts
    return float(contacts), float(buried_sasa), int(clashes)

def _composite_score(contacts: float, buried_sasa: float, clashes: int) -> float:
    return contacts + 0.02 * buried_sasa - 3.0 * clashes


def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    seeds = [17, 23, 29]
    targets = [None, 256, 64]
    configs: list[tuple[dict, list[str]]] = []
    for tgt, seed in zip(targets, seeds):
        variant = yaml.safe_load(yaml.safe_dump(input_dict))  # deep copy
        if msa_dir is not None and tgt is not None:
            for seq in variant.get("sequences", []):
                if "protein" in seq:
                    msa_rel = seq["protein"].get("msa")
                    new_msa = _make_subsample(msa_rel, msa_dir, f"sub{tgt}", tgt)
                    if new_msa:
                        seq["protein"]["msa"] = new_msa
        cli = ["--diffusion_samples", "6", "--seed", str(seed)]
        configs.append((variant, cli))
    return configs
    #cli_args = ["--diffusion_samples", "5"]
    #return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    seeds = [11, 13, 19]
    targets = [None, 256, 64]
    configs: list[tuple[dict, List[str]]] = []
    for tgt, seed in zip(targets, seeds):
        variant = yaml.safe_load(yaml.safe_dump(input_dict))
        if msa_dir is not None and tgt is not None:
            for seq in variant.get("sequences", []):
                if "protein" in seq:
                    msa_rel = seq["protein"].get("msa")
                    new_msa = _make_subsample(msa_rel, msa_dir, f"sub{tgt}", tgt)
                    if new_msa:
                        seq["protein"]["msa"] = new_msa
        cli = ["--diffusion_samples", "5", "--seed", str(seed)]
        configs.append((variant, cli))
    return configs
    #cli_args = ["--diffusion_samples", "5"]
    #return [(input_dict, cli_args)]

def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    #all_pdbs = []
    #for prediction_dir in prediction_dirs:
    #    config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
    #    all_pdbs.extend(config_pdbs)

    ## Sort all PDBs and return their paths
    #all_pdbs = sorted(all_pdbs)
    #return all_pdbs
    all_pdbs: list[Path] = []
    for d in prediction_dirs:
        all_pdbs.extend(_get_models_in_dir(d, datapoint.datapoint_id))
    if not all_pdbs:
        return []

    # cheap prefilter by contacts
    pre = []
    for p in all_pdbs:
        try:
            s = _load_structure(p)
            c, _, _ = _interface_score_complex(s)
            pre.append((c, p))
        except Exception:
            pre.append((-1, p))
    pre.sort(key=lambda x: x[0], reverse=True)
    shortlist = [p for _, p in pre[:max(20, min(50, len(all_pdbs)))]]

    scored = []
    for p in shortlist:
        try:
            s = _load_structure(p)
            c, b, cl = _interface_score_complex(s)
            scored.append((_composite_score(c, b, cl), p))
        except Exception:
            scored.append((-1e9, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    #all_pdbs = []
    #for prediction_dir in prediction_dirs:
    #    config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
    #    all_pdbs.extend(config_pdbs)
    #
    ## Sort all PDBs and return their paths
    #all_pdbs = sorted(all_pdbs)
    #return all_pdbs
    all_pdbs: list[Path] = []
    for d in prediction_dirs:
        all_pdbs.extend(_get_models_in_dir(d, datapoint.datapoint_id))
    if not all_pdbs:
        return []

    pre = []
    for p in all_pdbs:
        try:
            s = _load_structure(p)
            c, _, _ = _interface_score_ligand(s)
            pre.append((c, p))
        except Exception:
            pre.append((-1, p))
    pre.sort(key=lambda x: x[0], reverse=True)
    shortlist = [p for _, p in pre[:max(20, min(50, len(all_pdbs)))]]

    scored = []
    for p in shortlist:
        try:
            s = _load_structure(p)
            c, b, cl = _interface_score_ligand(s)
            scored.append((_composite_score(c, b, cl), p))
        except Exception:
            scored.append((-1e9, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--output_format", "pdb",
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
