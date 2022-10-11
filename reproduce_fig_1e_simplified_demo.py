import figures

if __name__ == "__main__":
    print("Creating figure 1e with FastTree instead of PhyML ...")

    figures.fig_lg_paper(
        evaluation_phylogeny_estimator_name="FastTree",
        output_image_dir="../../results/fig_1e_simplified/",
        num_processes=32,
    )  # Fig. 1e, 'fast' version with FastTree instead of PhyML.
    # < ~ 4 hr

    print("Creating figure 1e done!")
