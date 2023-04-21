import figures

################################################################################
##### Optional: set these to the locations of the provided simulated data ######
##### directories to skip simulating entirely. The simulated data is      ######
##### available at https://zenodo.org/record/7814723#.ZDSUDyZlBXk         ######
################################################################################

simulated_data_dirs_fig_1d = None
# Optional: Uncomment lines below, pointing to the downloaded data.
# simulated_data_dirs_fig_1d = {
#     "msa_dir": None,  # Optional: Fill this out
#     "gt_tree_dir": None,  # Optional: Fill this out
#     "gt_site_rates_dir": None,  # Optional: Fill this out
#     "gt_likelihood_dir": None,  # Optional: Fill this out
# }

simulated_data_dirs_fig_2ab = None
# Optional: Uncomment lines below, pointing to the downloaded data.
# simulated_data_dirs_fig_2ab = {
#     "msa_dir": None,  # Optional: Fill this out
#     "contact_map_dir": None,  # Optional: Fill this out
#     "gt_tree_dir": None,  # Optional: Fill this out
#     "gt_site_rates_dir": None,  # Optional: Fill this out
#     "gt_likelihood_dir": None,  # Optional: Fill this out
# }

simulated_data_dirs_fig_1bc = None
# Optional: Uncomment lines below, pointing to the downloaded data.
# simulated_data_dirs_fig_1bc = {
#     "msa_dir": None,  # Optional: Fill this out
#     "gt_tree_dir": None,  # Optional: Fill this out
#     "gt_site_rates_dir": None,  # Optional: Fill this out
#     "gt_likelihood_dir": None,  # Optional: Fill this out
#     "families_all.txt": None,  # Optional: Fill this out
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
