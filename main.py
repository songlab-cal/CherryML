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

    figures.fig_lg_paper()  # Fig. 1e
    # ~ 4 hr

    figures.fig_computational_and_stat_eff_cherry_vs_em()  # Fig. 1b, 1c
    # ~ 62 hr

    print("Creating figures done!")
