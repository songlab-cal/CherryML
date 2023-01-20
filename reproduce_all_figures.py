import figures

if __name__ == "__main__":
    print("Creating figures ...")

    figures.fig_single_site_quantization_error()  # Fig. 1d
    # ~ 8hr

    figures.fig_pair_site_quantization_error(
        Q_2_name="unmasked-co-transitions",
    )  # Fig. 2a
    # ~ 9hr

    figures.fig_pair_site_quantization_error(
        Q_2_name="unmasked-single-transitions",
    )  # Fig. 2b
    # ~ 30 min

    figures.fig_site_rates_vs_number_of_contacts()  # Fig. 2e
    # ~ 9 hr

    figures.fig_coevolution_vs_indep()  # Fig. 2c, 2d
    # ~ 1 min

    figures.fig_MSA_VI_cotransition()  # Comment in paragraph.
    # ~ 1 min

    figures.fig_lg_paper(num_processes=8)  # Fig. 1e
    # ~ 20 hr

    # Supp Fig.
    figures.fig_qmaker(clade_name="plant")
    figures.fig_qmaker(clade_name="insect")
    figures.fig_qmaker(clade_name="bird")
    figures.fig_qmaker(clade_name="mammal")
    # ~ a couple days

    print("Creating figures done!")
