import figures

################################################################################
##### Optional: set these to the locations of the provided simulated data ######
##### directories to skip simulating data entirely.                       ######
################################################################################

simulated_data_dirs_fig_1d = None
# simulated_data_dirs_fig_1d = {
#     "msa_dir": "_cache_benchmarking/subset_msa_to_leaf_nodes/b8523c83b3e5478556b367a5f5bf5428e9f3244ce206f180f40854b716a4a6bd/output_msa_dir",
#     "gt_tree_dir": "_cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_tree_dir",
#     "gt_site_rates_dir": "_cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_site_rates_dir",
#     "gt_likelihood_dir": "_cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_likelihood_dir",
# }

simulated_data_dirs_fig_2ab = None
# simulated_data_dirs_fig_2ab = {
#     "msa_dir": "cache_benchmarking/subset_msa_to_leaf_nodes/08796a1c5ca4c6070c77b91780889748a6560723d3d88229a8af5add02539bd7/output_msa_dir",
#     "contact_map_dir": "cache_benchmarking/create_maximal_matching_contact_map/492b760db9307f08efecec9ba203d84cdccecbe80ec8007d25580d595b342f7c/o_contact_map_dir",
#     "gt_tree_dir": "cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_tree_dir",
#     "gt_site_rates_dir": "cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_site_rates_dir",
#     "gt_likelihood_dir": "cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_likelihood_dir",
# }

simulated_data_dirs_fig_1bc = None
# simulated_data_dirs_fig_1bc = {
#     "msa_dir": "_cache_benchmarking_em/subset_msa_to_leaf_nodes/da6ef3b0d58b16b6b9f5ad5628ffe750fc41b6b444a12b32f4f5d904a220facf/output_msa_dir",
#     "gt_tree_dir": "_cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_tree_dir",
#     "gt_site_rates_dir": "_cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_site_rates_dir",
#     "gt_likelihood_dir": "_cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_likelihood_dir",
#     "families_all": "fig_1bc_simulated_data_families_all.txt",
# }

################################################################################
################################# main #########################################
################################################################################

if __name__ == "__main__":
    print("Creating figures ...")

    figures.fig_single_site_quantization_error(
        simulated_data_dirs=simulated_data_dirs_fig_1d,
    )  # Fig. 1d
    # ~ 8hr

    figures.fig_pair_site_quantization_error(
        Q_2_name="unmasked-co-transitions",
        simulated_data_dirs=simulated_data_dirs_fig_2ab,
    )  # Fig. 2a
    # ~ 9hr

    figures.fig_pair_site_quantization_error(
        Q_2_name="unmasked-single-transitions",
        simulated_data_dirs=simulated_data_dirs_fig_2ab,
    )  # Fig. 2b
    # ~ 30 min

    figures.fig_site_rates_vs_number_of_contacts()  # Fig. 2e
    # ~ 9 hr

    figures.fig_coevolution_vs_indep()  # Fig. 2c, 2d
    # ~ 1 min

    figures.fig_MSA_VI_cotransition()  # Comment in paragraph.
    # ~ 1 min

    figures.fig_lg_paper()  # Fig. 1e
    # ~ 20 hr

    figures.fig_computational_and_stat_eff_cherry_vs_em(
        simulated_data_dirs=simulated_data_dirs_fig_1bc,
    )  # Fig. 1b, 1c
    # ~ 62 hr

    # Supp Fig.
    figures.fig_qmaker(clade_name="plant")
    figures.fig_qmaker(clade_name="insect")
    figures.fig_qmaker(clade_name="bird")
    figures.fig_qmaker(clade_name="mammal")
    # ~ 3 days

    print("Creating figures done!")
