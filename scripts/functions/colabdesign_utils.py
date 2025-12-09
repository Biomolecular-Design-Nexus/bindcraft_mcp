####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax
from loguru import logger
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages
from .pyrosetta_utils import pr_relax, align_pdbs
from .generic_utils import update_failures

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, failure_csv):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    logger.info(f"[AF2] Initializing hallucination model (multimer={advanced_settings['use_multimer_design']})")
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"],
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"], rm_target_sc=advanced_settings["rm_template_sc_design"])

    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"],"cutoff":advanced_settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"],"cutoff":advanced_settings["inter_contact_distance"],"binary":False})


    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    logger.info(f"[AF2] Running {advanced_settings['design_algorithm']} design algorithm")

    if advanced_settings["design_algorithm"] == '2stage':
        af_model.design_pssm_semigreedy(soft_iters=advanced_settings["soft_iterations"], hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models,
                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

    elif advanced_settings["design_algorithm"] == '3stage':
        af_model.design_3stage(soft_iters=advanced_settings["soft_iterations"], temp_iters=advanced_settings["temporary_iterations"], hard_iters=advanced_settings["hard_iterations"],
                                num_models=1, models=design_models, sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'greedy':
        af_model.design_semigreedy(advanced_settings["greedy_iterations"], tries=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == 'mcmc':
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    elif advanced_settings["design_algorithm"] == '4stage':
        # Stage 1: initial logits
        logger.info("[AF2] Stage 1/4: Logits optimization")
        af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)
        initial_plddt = get_best_plddt(af_model, length)

        if initial_plddt > 0.65:
            if advanced_settings["optimise_beta"]:
                af_model.save_pdb(model_pdb_path)
                _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
                os.remove(model_pdb_path)
                if float(beta) > 15:
                    advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                    advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                    af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                    logger.info(f"[AF2] Beta sheet detected ({beta:.1f}%), adjusting iterations")

            logits_iter = advanced_settings["soft_iterations"] - 50
            if logits_iter > 0:
                af_model.clear_best()
                af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                    ramp_recycles=False, save_best=True)
                af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
                logit_plddt = get_best_plddt(af_model, length)
            else:
                logit_plddt = initial_plddt

            # Stage 2: softmax
            if advanced_settings["temporary_iterations"] > 0:
                logger.info("[AF2] Stage 2/4: Softmax optimization")
                af_model.clear_best()
                af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
                softmax_plddt = get_best_plddt(af_model, length)
            else:
                softmax_plddt = logit_plddt

            # Stage 3: one-hot
            if softmax_plddt > 0.65:
                if advanced_settings["hard_iterations"] > 0:
                    logger.info("[AF2] Stage 3/4: One-hot optimization")
                    af_model.clear_best()
                    af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                    sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                    onehot_plddt = get_best_plddt(af_model, length)

                # Stage 4: greedy
                if onehot_plddt > 0.65:
                    if advanced_settings["greedy_iterations"] > 0:
                        logger.info("[AF2] Stage 4/4: PSSM semigreedy optimization")
                        af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models,
                                                        num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)
                else:
                    update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    logger.warning(f"[AF2] One-hot pLDDT too low: {onehot_plddt:.2f}")
            else:
                update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                logger.warning(f"[AF2] Softmax pLDDT too low: {softmax_plddt:.2f}")
        else:
            update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            logger.warning(f"[AF2] Initial pLDDT too low: {initial_plddt:.2f}")

    else:
        logger.error(f"[AF2] Invalid design algorithm: {advanced_settings['design_algorithm']}")
        exit()
        return

    ### save trajectory PDB
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # Validate trajectory
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        update_failures(failure_csv, 'Trajectory_Clashes')
        logger.warning(f"[AF2] Clashes detected ({ca_clashes}), skipping MPNN")
    elif final_plddt < 0.7:
        af_model.aux["log"]["terminate"] = "LowConfidence"
        update_failures(failure_csv, 'Trajectory_final_pLDDT')
        logger.warning(f"[AF2] pLDDT too low ({final_plddt:.2f}), skipping MPNN")
    else:
        binder_contacts = hotspot_residues(model_pdb_path)
        binder_contacts_n = len(binder_contacts.items())
        if binder_contacts_n < 3:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            update_failures(failure_csv, 'Trajectory_Contacts')
            logger.warning(f"[AF2] Too few contacts ({binder_contacts_n}), skipping MPNN")
        else:
            af_model.aux["log"]["terminate"] = ""
            logger.info(f"[AF2] Trajectory OK: pLDDT={final_plddt:.2f}, contacts={binder_contacts_n}")

    # move low quality prediction
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    # Save outputs
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, failure_csv, seed=None):
    prediction_stats = {}
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    pass_af2_filters = True
    filter_failures = {}

    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"])

            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'i_pTM': round(prediction_metrics['i_ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2),
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats

            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]

            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

            if not pass_af2_filters:
                logger.warning(f"[AF2] Prediction filters failed for {mpnn_design_name}")
                break

    if filter_failures:
        update_failures(failure_csv, filter_failures)

    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            pr_relax(complex_pdb, mpnn_relaxed)
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)

    return prediction_stats, pass_af2_filters

# run prediction for binder alone
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None):
    binder_stats = {}

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)

    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae

            # align binder model to trajectory binder
            align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2),
                'pTM': round(prediction_metrics['ptm'], 2),
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats

    return binder_stats

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    clear_mem()

    logger.info(f"[MPNN] Initializing model (noise={advanced_settings['backbone_noise']})")
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        logger.info(f"[MPNN] Fixing interface residues: {trajectory_interface_residues}")
    else:
        fixed_positions = 'A'

    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    logger.info(f"[MPNN] Sampling {advanced_settings['num_seqs']} sequences (temp={advanced_settings['sampling_temp']})")
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)

            # Close the figure
            plt.close()
