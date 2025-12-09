#!/usr/bin/env python
"""
BindCraft Binder Design Script

A standalone script for running BindCraft binder design workflow.
This script imports all necessary functions from the BindCraft repository
and provides a clean interface for protein binder design.

Usage:
    python run_bindcraft.py --settings target.json --filters filters.json --advanced advanced.json

Example:
    python run_bindcraft.py \
        --settings examples/PDL1/target.json \
        --filters examples/PDL1/default_filters.json \
        --advanced examples/PDL1/default_4stage_multimer.json
"""

import os
import sys
import time
import gc
import argparse
import shutil
import numpy as np
import pandas as pd
from loguru import logger

# Script directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure loguru logger
def setup_logging(log_dir=None, log_level="INFO"):
    """Configure loguru logging with file and console outputs."""
    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Add file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "bindcraft_{time:YYYY-MM-DD_HH-mm-ss}.log")
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",
            retention="7 days"
        )
        logger.info(f"Log file created at: {log_dir}")

    return logger

# Import all BindCraft functions from local functions module
from functions import *

# Import PyRosetta
import pyrosetta as pr


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run BindCraft binder design.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bindcraft.py -s target.json -f filters.json -a advanced.json
  python run_bindcraft.py --settings examples/PDL1/target.json
        """
    )

    parser.add_argument(
        '--settings', '-s',
        type=str,
        required=True,
        help='Path to the target settings JSON file (required)'
    )
    parser.add_argument(
        '--filters', '-f',
        type=str,
        default=None,
        help='Path to the filters JSON file for design filtering'
    )
    parser.add_argument(
        '--advanced', '-a',
        type=str,
        default=None,
        help='Path to the advanced settings JSON file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def setup_default_paths(args):
    """Set up default paths for filters and advanced settings if not provided."""
    # Look for defaults in repo/BindCraft if not provided
    repo_bindcraft_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "repo", "BindCraft")

    # Default filter settings
    if args.filters is None:
        args.filters = os.path.join(repo_bindcraft_dir, 'settings_filters', 'default_filters.json')

    # Default advanced settings
    if args.advanced is None:
        args.advanced = os.path.join(repo_bindcraft_dir, 'settings_advanced', 'default_4stage_multimer.json')

    return args


def initialize_pyrosetta(advanced_settings):
    """Initialize PyRosetta with appropriate settings."""
    logger.info("Initializing PyRosetta...")
    logger.debug(f"DAlphaBall path: {advanced_settings['dalphaball_path']}")

    pr.init(
        f'-ignore_unrecognized_res '
        f'-ignore_zero_occupancy '
        f'-mute all '
        f'-holes:dalphaball {advanced_settings["dalphaball_path"]} '
        f'-corrections::beta_nov16 true '
        f'-relax:default_repeats 1'
    )
    logger.success("PyRosetta initialized successfully")


def format_time(seconds):
    """Format time in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, {secs} seconds"


def run_bindcraft(target_settings, advanced_settings, filters,
                  design_models, prediction_models, multimer_validation,
                  design_paths, trajectory_csv, mpnn_csv, final_csv, failure_csv,
                  settings_file, filters_file, advanced_file):
    """
    Main BindCraft design loop.

    Args:
        target_settings: Target configuration dictionary
        advanced_settings: Advanced settings dictionary
        filters: Filter thresholds dictionary
        design_models: List of AF2 model indices for design
        prediction_models: List of AF2 model indices for prediction
        multimer_validation: Whether to use multimer for validation
        design_paths: Dictionary of output paths
        trajectory_csv: Path to trajectory stats CSV
        mpnn_csv: Path to MPNN design stats CSV
        final_csv: Path to final design stats CSV
        failure_csv: Path to failure tracking CSV
        settings_file: Name of settings file (for logging)
        filters_file: Name of filters file (for logging)
        advanced_file: Name of advanced settings file (for logging)
    """
    # Get label definitions
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    # Initialize counters
    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0

    logger.info("=" * 60)
    logger.info("Starting BindCraft Binder Design")
    logger.info("=" * 60)
    logger.info(f"Target: {settings_file}")
    logger.info(f"Design settings: {advanced_file}")
    logger.info(f"Filter settings: {filters_file}")
    logger.info(f"Design algorithm: {advanced_settings['design_algorithm']}")
    logger.info(f"Binder length range: {target_settings['lengths']}")
    logger.info(f"Target designs: {target_settings['number_of_final_designs']}")
    logger.info(f"AF2 design models: {design_models}")
    logger.info(f"AF2 prediction models: {prediction_models}")
    logger.debug(f"Output directory: {target_settings['design_path']}")

    # Main design loop
    while True:
        # Check if target number of binders reached
        final_designs_reached = check_accepted_designs(
            design_paths, mpnn_csv, final_labels, final_csv,
            advanced_settings, target_settings, design_labels
        )

        if final_designs_reached:
            logger.success(f"Target number of designs reached!")
            break

        # Check if maximum trajectories reached
        max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

        if max_trajectories_reached:
            logger.warning("Maximum trajectories reached, stopping...")
            break

        # Start trajectory timing
        trajectory_start_time = time.time()

        # Generate random seed
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        # Sample binder length from distribution
        samples = np.arange(
            min(target_settings["lengths"]),
            max(target_settings["lengths"]) + 1
        )
        length = np.random.choice(samples)

        # Load helicity value
        helicity_value = load_helicity(advanced_settings)

        # Generate design name and check for existing trajectory
        design_name = f"{target_settings['binder_name']}_l{length}_s{seed}"
        trajectory_dirs = [
            "Trajectory", "Trajectory/Relaxed",
            "Trajectory/LowConfidence", "Trajectory/Clashing"
        ]
        trajectory_exists = any(
            os.path.exists(os.path.join(design_paths[d], f"{design_name}.pdb"))
            for d in trajectory_dirs
        )

        if not trajectory_exists:
            logger.info("-" * 50)
            logger.info(f"Trajectory #{trajectory_n}: {design_name}")
            logger.info(f"Length: {length}, Seed: {seed}, Helicity: {helicity_value}")

            # Run binder hallucination
            logger.info("Running AF2 binder hallucination...")
            trajectory = binder_hallucination(
                design_name,
                target_settings["starting_pdb"],
                target_settings["chains"],
                target_settings["target_hotspot_residues"],
                length,
                seed,
                helicity_value,
                design_models,
                advanced_settings,
                design_paths,
                failure_csv
            )

            trajectory_metrics = copy_dict(trajectory._tmp["best"]["aux"]["log"])
            trajectory_pdb = os.path.join(design_paths["Trajectory"], f"{design_name}.pdb")

            # Round metrics
            trajectory_metrics = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in trajectory_metrics.items()
            }

            # Log trajectory time and metrics
            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = format_time(trajectory_time)
            logger.info(f"AF2 Hallucination completed in {trajectory_time_text}")
            logger.debug(f"Trajectory metrics: pLDDT={trajectory_metrics.get('plddt')}, "
                        f"pTM={trajectory_metrics.get('ptm')}, i_pTM={trajectory_metrics.get('i_ptm')}")

            # Proceed if no termination signal
            if trajectory.aux["log"]["terminate"] == "":
                # Relax trajectory
                logger.info("Running PyRosetta relaxation...")
                trajectory_relaxed = os.path.join(
                    design_paths["Trajectory/Relaxed"], f"{design_name}.pdb"
                )
                pr_relax(trajectory_pdb, trajectory_relaxed)
                logger.debug(f"Relaxed structure saved to: {trajectory_relaxed}")

                binder_chain = "B"

                # Calculate clashes
                logger.debug("Calculating clash scores...")
                num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
                num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)
                logger.debug(f"Clashes - Unrelaxed: {num_clashes_trajectory}, Relaxed: {num_clashes_relaxed}")

                # Secondary structure content
                logger.debug("Calculating secondary structure content...")
                (trajectory_alpha, trajectory_beta, trajectory_loops,
                 trajectory_alpha_interface, trajectory_beta_interface,
                 trajectory_loops_interface, trajectory_i_plddt,
                 trajectory_ss_plddt) = calc_ss_percentage(
                    trajectory_pdb, advanced_settings, binder_chain
                )
                logger.debug(f"Secondary structure - Helix: {trajectory_alpha}%, Sheet: {trajectory_beta}%, Loop: {trajectory_loops}%")

                # Interface scores
                logger.info("Calculating PyRosetta interface scores...")
                (trajectory_interface_scores, trajectory_interface_AA,
                 trajectory_interface_residues) = score_interface(
                    trajectory_relaxed, binder_chain
                )
                logger.debug(f"Interface dG: {trajectory_interface_scores['interface_dG']}, "
                            f"SC: {trajectory_interface_scores['interface_sc']}")

                # Get sequence
                trajectory_sequence = trajectory.get_seq(get_best=True)[0]

                # Validate sequence
                traj_seq_notes = validate_design_sequence(
                    trajectory_sequence, num_clashes_relaxed, advanced_settings
                )

                # Target RMSD
                trajectory_target_rmsd = target_pdb_rmsd(
                    trajectory_pdb,
                    target_settings["starting_pdb"],
                    target_settings["chains"]
                )

                # Save trajectory statistics
                trajectory_data = [
                    design_name, advanced_settings["design_algorithm"], length,
                    seed, helicity_value, target_settings["target_hotspot_residues"],
                    trajectory_sequence, trajectory_interface_residues,
                    trajectory_metrics['plddt'], trajectory_metrics['ptm'],
                    trajectory_metrics['i_ptm'], trajectory_metrics['pae'],
                    trajectory_metrics['i_pae'], trajectory_i_plddt, trajectory_ss_plddt,
                    num_clashes_trajectory, num_clashes_relaxed,
                    trajectory_interface_scores['binder_score'],
                    trajectory_interface_scores['surface_hydrophobicity'],
                    trajectory_interface_scores['interface_sc'],
                    trajectory_interface_scores['interface_packstat'],
                    trajectory_interface_scores['interface_dG'],
                    trajectory_interface_scores['interface_dSASA'],
                    trajectory_interface_scores['interface_dG_SASA_ratio'],
                    trajectory_interface_scores['interface_fraction'],
                    trajectory_interface_scores['interface_hydrophobicity'],
                    trajectory_interface_scores['interface_nres'],
                    trajectory_interface_scores['interface_interface_hbonds'],
                    trajectory_interface_scores['interface_hbond_percentage'],
                    trajectory_interface_scores['interface_delta_unsat_hbonds'],
                    trajectory_interface_scores['interface_delta_unsat_hbonds_percentage'],
                    trajectory_alpha_interface, trajectory_beta_interface,
                    trajectory_loops_interface, trajectory_alpha, trajectory_beta,
                    trajectory_loops, trajectory_interface_AA, trajectory_target_rmsd,
                    trajectory_time_text, traj_seq_notes,
                    settings_file, filters_file, advanced_file
                ]
                insert_data(trajectory_csv, trajectory_data)

                # Skip MPNN if no interface residues
                if not trajectory_interface_residues:
                    logger.warning(f"No interface residues found for {design_name}, skipping MPNN optimization")
                    trajectory_n += 1
                    gc.collect()
                    continue

                # MPNN redesign
                if advanced_settings["enable_mpnn"]:
                    logger.info("Starting MPNN sequence optimization...")
                    accepted_mpnn = process_mpnn_sequences(
                        trajectory, trajectory_pdb, design_name, length, seed,
                        helicity_value, binder_chain, trajectory_interface_residues,
                        trajectory_beta, target_settings, advanced_settings, filters,
                        design_models, prediction_models, multimer_validation,
                        design_paths, design_labels, final_labels,
                        mpnn_csv, final_csv, failure_csv,
                        settings_file, filters_file, advanced_file
                    )
                    accepted_designs += accepted_mpnn

                    # Remove unrelaxed trajectory PDB if configured
                    if advanced_settings["remove_unrelaxed_trajectory"]:
                        if os.path.exists(trajectory_pdb):
                            os.remove(trajectory_pdb)

                # Monitor rejection rate
                if (trajectory_n >= advanced_settings["start_monitoring"] and
                    advanced_settings["enable_rejection_check"]):
                    acceptance = accepted_designs / trajectory_n
                    if acceptance < advanced_settings["acceptance_rate"]:
                        logger.error("The ratio of successful designs is lower than defined acceptance rate!")
                        logger.error("Consider changing your design settings!")
                        logger.error("Script execution stopping...")
                        break

        trajectory_n += 1
        gc.collect()

    # Final summary
    elapsed_time = time.time() - script_start_time
    elapsed_text = format_time(elapsed_time)
    logger.info("=" * 60)
    logger.success(f"BindCraft Design Complete!")
    logger.info(f"Total trajectories: {trajectory_n}")
    logger.info(f"Accepted designs: {accepted_designs}")
    logger.info(f"Total time: {elapsed_text}")
    logger.info("=" * 60)


def process_mpnn_sequences(trajectory, trajectory_pdb, design_name, length, seed,
                           helicity_value, binder_chain, trajectory_interface_residues,
                           trajectory_beta, target_settings, advanced_settings, filters,
                           design_models, prediction_models, multimer_validation,
                           design_paths, design_labels, final_labels,
                           mpnn_csv, final_csv, failure_csv,
                           settings_file, filters_file, advanced_file):
    """
    Process MPNN sequence redesign for a trajectory.

    Returns:
        int: Number of accepted MPNN designs
    """
    mpnn_n = 1
    accepted_mpnn = 0
    mpnn_dict = {}
    design_start_time = time.time()

    # Generate MPNN sequences
    logger.info("Generating MPNN sequences...")
    mpnn_trajectories = mpnn_gen_sequence(
        trajectory_pdb, binder_chain,
        trajectory_interface_residues, advanced_settings
    )
    logger.debug(f"Generated {advanced_settings['num_seqs']} MPNN sequences")

    existing_mpnn_sequences = set(
        pd.read_csv(mpnn_csv, usecols=['Sequence'])['Sequence'].values
    )

    # Filter sequences
    restricted_AAs = set(
        aa.strip().upper()
        for aa in advanced_settings["omit_AAs"].split(',')
    ) if advanced_settings["force_reject_AA"] else set()

    mpnn_sequences = sorted({
        mpnn_trajectories['seq'][n][-length:]: {
            'seq': mpnn_trajectories['seq'][n][-length:],
            'score': mpnn_trajectories['score'][n],
            'seqid': mpnn_trajectories['seqid'][n]
        } for n in range(advanced_settings["num_seqs"])
        if (not restricted_AAs or not any(
            aa in mpnn_trajectories['seq'][n][-length:].upper()
            for aa in restricted_AAs
        ))
        and mpnn_trajectories['seq'][n][-length:] not in existing_mpnn_sequences
    }.values(), key=lambda x: x['score'])

    del existing_mpnn_sequences

    if not mpnn_sequences:
        logger.warning('Duplicate MPNN designs sampled, skipping current trajectory optimization')
        return 0

    logger.info(f"Processing {len(mpnn_sequences)} unique MPNN sequences")

    # Optimize for beta sheets if needed
    if advanced_settings["optimise_beta"] and float(trajectory_beta) > 15:
        advanced_settings["num_recycles_validation"] = advanced_settings["optimise_beta_recycles_valid"]
        logger.debug(f"Beta sheet detected, increasing validation recycles to {advanced_settings['num_recycles_validation']}")

    # Compile prediction models
    logger.info("Compiling AF2 prediction models...")
    clear_mem()

    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=advanced_settings["predict_initial_guess"],
        use_initial_atom_pos=advanced_settings["predict_bigbang"]
    )
    logger.debug("Complex prediction model compiled")

    if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:
        complex_prediction_model.prep_inputs(
            pdb_filename=trajectory_pdb, chain='A', binder_chain='B',
            binder_len=length, use_binder_template=True,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"],
            rm_template_ic=True
        )
    else:
        complex_prediction_model.prep_inputs(
            pdb_filename=target_settings["starting_pdb"],
            chain=target_settings["chains"], binder_len=length,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"]
        )

    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination", use_templates=False,
        initial_guess=False, use_initial_atom_pos=False,
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation
    )
    binder_prediction_model.prep_inputs(length=length)
    logger.debug("Binder monomer prediction model compiled")

    # Process each MPNN sequence
    for mpnn_sequence in mpnn_sequences:
        mpnn_time = time.time()

        mpnn_design_name = f"{design_name}_mpnn{mpnn_n}"
        mpnn_score = round(mpnn_sequence['score'], 2)
        mpnn_seqid = round(mpnn_sequence['seqid'], 2)

        logger.info(f"Processing MPNN sequence {mpnn_n}/{len(mpnn_sequences)}: {mpnn_design_name}")
        logger.debug(f"MPNN score: {mpnn_score}, Seq recovery: {mpnn_seqid}")

        mpnn_dict[mpnn_design_name] = {
            'seq': mpnn_sequence['seq'],
            'score': mpnn_score,
            'seqid': mpnn_seqid
        }

        # Save FASTA if configured
        if advanced_settings["save_mpnn_fasta"]:
            save_fasta(mpnn_design_name, mpnn_sequence['seq'], design_paths)

        # Predict complex
        logger.info("Running AF2 complex prediction...")
        mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(
            complex_prediction_model, mpnn_sequence['seq'], mpnn_design_name,
            target_settings["starting_pdb"], target_settings["chains"],
            length, trajectory_pdb, prediction_models, advanced_settings,
            filters, design_paths, failure_csv
        )

        if not pass_af2_filters:
            logger.warning(f"AF2 filters not passed for {mpnn_design_name}, skipping interface scoring")
            mpnn_n += 1
            continue

        logger.debug(f"AF2 complex prediction passed filters")

        # Calculate per-model statistics
        mpnn_interface_residues = None
        for model_num in prediction_models:
            mpnn_design_pdb = os.path.join(
                design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb"
            )
            mpnn_design_relaxed = os.path.join(
                design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb"
            )

            if os.path.exists(mpnn_design_pdb):
                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = \
                    score_interface(mpnn_design_relaxed, binder_chain)

                (mpnn_alpha, mpnn_beta, mpnn_loops, mpnn_alpha_interface,
                 mpnn_beta_interface, mpnn_loops_interface, mpnn_i_plddt,
                 mpnn_ss_plddt) = calc_ss_percentage(
                    mpnn_design_pdb, advanced_settings, binder_chain
                )

                rmsd_site = unaligned_rmsd(
                    trajectory_pdb, mpnn_design_pdb, binder_chain, binder_chain
                )
                target_rmsd = target_pdb_rmsd(
                    mpnn_design_pdb, target_settings["starting_pdb"],
                    target_settings["chains"]
                )

                mpnn_complex_statistics[model_num+1].update({
                    'i_pLDDT': mpnn_i_plddt,
                    'ss_pLDDT': mpnn_ss_plddt,
                    'Unrelaxed_Clashes': num_clashes_mpnn,
                    'Relaxed_Clashes': num_clashes_mpnn_relaxed,
                    'Binder_Energy_Score': mpnn_interface_scores['binder_score'],
                    'Surface_Hydrophobicity': mpnn_interface_scores['surface_hydrophobicity'],
                    'ShapeComplementarity': mpnn_interface_scores['interface_sc'],
                    'PackStat': mpnn_interface_scores['interface_packstat'],
                    'dG': mpnn_interface_scores['interface_dG'],
                    'dSASA': mpnn_interface_scores['interface_dSASA'],
                    'dG/dSASA': mpnn_interface_scores['interface_dG_SASA_ratio'],
                    'Interface_SASA_%': mpnn_interface_scores['interface_fraction'],
                    'Interface_Hydrophobicity': mpnn_interface_scores['interface_hydrophobicity'],
                    'n_InterfaceResidues': mpnn_interface_scores['interface_nres'],
                    'n_InterfaceHbonds': mpnn_interface_scores['interface_interface_hbonds'],
                    'InterfaceHbondsPercentage': mpnn_interface_scores['interface_hbond_percentage'],
                    'n_InterfaceUnsatHbonds': mpnn_interface_scores['interface_delta_unsat_hbonds'],
                    'InterfaceUnsatHbondsPercentage': mpnn_interface_scores['interface_delta_unsat_hbonds_percentage'],
                    'InterfaceAAs': mpnn_interface_AA,
                    'Interface_Helix%': mpnn_alpha_interface,
                    'Interface_BetaSheet%': mpnn_beta_interface,
                    'Interface_Loop%': mpnn_loops_interface,
                    'Binder_Helix%': mpnn_alpha,
                    'Binder_BetaSheet%': mpnn_beta,
                    'Binder_Loop%': mpnn_loops,
                    'Hotspot_RMSD': rmsd_site,
                    'Target_RMSD': target_rmsd
                })

                if advanced_settings["remove_unrelaxed_complex"]:
                    os.remove(mpnn_design_pdb)

        # Calculate averages
        mpnn_complex_averages = calculate_averages(mpnn_complex_statistics, handle_aa=True)
        logger.debug(f"Complex averages - pLDDT: {mpnn_complex_averages.get('pLDDT')}, "
                    f"i_pTM: {mpnn_complex_averages.get('i_pTM')}")

        # Predict binder alone
        logger.info("Running AF2 binder monomer prediction...")
        binder_statistics = predict_binder_alone(
            binder_prediction_model, mpnn_sequence['seq'], mpnn_design_name,
            length, trajectory_pdb, binder_chain, prediction_models,
            advanced_settings, design_paths
        )

        for model_num in prediction_models:
            mpnn_binder_pdb = os.path.join(
                design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb"
            )

            if os.path.exists(mpnn_binder_pdb):
                rmsd_binder = unaligned_rmsd(
                    trajectory_pdb, mpnn_binder_pdb, binder_chain, "A"
                )
                binder_statistics[model_num+1].update({'Binder_RMSD': rmsd_binder})

                if advanced_settings["remove_binder_monomer"]:
                    os.remove(mpnn_binder_pdb)

        binder_averages = calculate_averages(binder_statistics)

        # Validate sequence
        seq_notes = validate_design_sequence(
            mpnn_sequence['seq'],
            mpnn_complex_averages.get('Relaxed_Clashes', None),
            advanced_settings
        )

        # Time measurement
        mpnn_end_time = time.time() - mpnn_time
        elapsed_mpnn_text = format_time(mpnn_end_time)

        # Build data row
        model_numbers = range(1, 6)
        statistics_labels = [
            'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT',
            'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score',
            'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat',
            'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity',
            'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
            'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage',
            'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
            'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs',
            'Hotspot_RMSD', 'Target_RMSD'
        ]

        mpnn_data = [
            mpnn_design_name, advanced_settings["design_algorithm"], length,
            seed, helicity_value, target_settings["target_hotspot_residues"],
            mpnn_sequence['seq'], mpnn_interface_residues, mpnn_score, mpnn_seqid
        ]

        for label in statistics_labels:
            mpnn_data.append(mpnn_complex_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(mpnn_complex_statistics.get(model, {}).get(label, None))

        for label in ['pLDDT', 'pTM', 'pAE', 'Binder_RMSD']:
            mpnn_data.append(binder_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

        mpnn_data.extend([elapsed_mpnn_text, seq_notes, settings_file, filters_file, advanced_file])

        insert_data(mpnn_csv, mpnn_data)

        # Find best model
        plddt_values = {i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None}
        highest_plddt_key = int(max(plddt_values, key=plddt_values.get))
        best_model_number = highest_plddt_key - 10
        best_model_pdb = os.path.join(
            design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{best_model_number}.pdb"
        )

        # Check filters
        filter_conditions = check_filters(mpnn_data, design_labels, filters)

        if filter_conditions == True:
            logger.success(f"{mpnn_design_name} PASSED all filters!")
            accepted_mpnn += 1

            shutil.copy(best_model_pdb, design_paths["Accepted"])
            logger.debug(f"Accepted design saved to: {design_paths['Accepted']}")

            final_data = [''] + mpnn_data
            insert_data(final_csv, final_data)

            if advanced_settings["save_design_animations"]:
                accepted_animation = os.path.join(
                    design_paths["Accepted/Animation"], f"{design_name}.html"
                )
                if not os.path.exists(accepted_animation):
                    src_animation = os.path.join(
                        design_paths["Trajectory/Animation"], f"{design_name}.html"
                    )
                    if os.path.exists(src_animation):
                        shutil.copy(src_animation, accepted_animation)

            plot_files = os.listdir(design_paths["Trajectory/Plots"])
            plots_to_copy = [
                f for f in plot_files
                if f.startswith(design_name) and f.endswith('.png')
            ]
            for accepted_plot in plots_to_copy:
                source_plot = os.path.join(design_paths["Trajectory/Plots"], accepted_plot)
                target_plot = os.path.join(design_paths["Accepted/Plots"], accepted_plot)
                if not os.path.exists(target_plot):
                    shutil.copy(source_plot, target_plot)
        else:
            logger.warning(f"{mpnn_design_name} FAILED filters: {filter_conditions[:3]}...")
            failure_df = pd.read_csv(failure_csv)
            special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
            incremented_columns = set()

            for column in filter_conditions:
                base_column = column
                for prefix in special_prefixes:
                    if column.startswith(prefix):
                        base_column = column.split('_', 1)[1]

                if base_column not in incremented_columns:
                    failure_df[base_column] = failure_df[base_column] + 1
                    incremented_columns.add(base_column)

            failure_df.to_csv(failure_csv, index=False)
            shutil.copy(best_model_pdb, design_paths["Rejected"])

        mpnn_n += 1

        if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
            break

    if accepted_mpnn >= 1:
        logger.success(f"Found {accepted_mpnn} MPNN designs passing filters for {design_name}")
    else:
        logger.warning(f"No accepted MPNN designs found for trajectory {design_name}")

    # Time summary
    design_time = time.time() - design_start_time
    design_time_text = format_time(design_time)
    logger.info(f"MPNN optimization for {design_name} completed in {design_time_text}")

    return accepted_mpnn


def main():
    """Main entry point for BindCraft."""
    # Parse arguments first to get log level
    args = parse_arguments()

    # Check GPU availability (this also logs GPU info)
    check_jax_gpu()

    # Setup default paths
    args = setup_default_paths(args)

    # Validate input files
    settings_path, filters_path, advanced_path = perform_input_check(args)

    # Load settings
    target_settings, advanced_settings, filters = load_json_settings(
        settings_path, filters_path, advanced_path
    )

    # Setup logging with output directory
    log_dir = os.path.join(target_settings["design_path"], "logs")
    setup_logging(log_dir=log_dir, log_level=args.log_level)

    logger.info("BindCraft Binder Design Pipeline")
    logger.info(f"Settings file: {settings_path}")
    logger.info(f"Filters file: {filters_path}")
    logger.info(f"Advanced settings: {advanced_path}")

    settings_file = os.path.basename(settings_path).split('.')[0]
    filters_file = os.path.basename(filters_path).split('.')[0]
    advanced_file = os.path.basename(advanced_path).split('.')[0]

    # Load AF2 models
    logger.info("Loading AlphaFold2 model configuration...")
    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )
    logger.debug(f"Design models: {design_models}")
    logger.debug(f"Prediction models: {prediction_models}")
    logger.debug(f"Multimer validation: {multimer_validation}")

    # Check advanced settings - use SCRIPT_DIR for local functions path
    logger.info("Validating advanced settings...")
    advanced_settings = perform_advanced_settings_check(advanced_settings, SCRIPT_DIR)
    logger.debug(f"AF2 params dir: {advanced_settings['af_params_dir']}")
    logger.debug(f"DSSP path: {advanced_settings['dssp_path']}")
    logger.debug(f"DAlphaBall path: {advanced_settings['dalphaball_path']}")

    # Generate output directories
    logger.info(f"Creating output directories in: {target_settings['design_path']}")
    design_paths = generate_directories(target_settings["design_path"])

    # Generate dataframes
    logger.info("Initializing tracking CSVs...")
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    # Initialize PyRosetta
    initialize_pyrosetta(advanced_settings)

    # Run main design loop
    logger.info("Starting main design loop...")
    run_bindcraft(
        target_settings, advanced_settings, filters,
        design_models, prediction_models, multimer_validation,
        design_paths, trajectory_csv, mpnn_csv, final_csv, failure_csv,
        settings_file, filters_file, advanced_file
    )


if __name__ == "__main__":
    main()
