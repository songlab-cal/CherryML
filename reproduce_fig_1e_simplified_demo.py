import figures

if __name__ == "__main__":
    print("Creating figure 1e with FastTree instead of PhyML ...")

    figures.fig_lg_paper(
        evaluation_phylogeny_estimator_name="FastTree",
        output_image_dir="fig_1e_simplified/",
        num_processes=32,
        rate_estimator_names=[
            ("reproduced WAG", "WAG\nrate matrix"),
            ("reproduced LG", "LG\nrate matrix"),
            ("Cherry++__4", "LG w/CherryML\n(re-estimated)"),
        ],
        lg_pfam_training_alignments_dir="data/lg_paper_data/lg_PfamTrainingAlignments",
        lg_pfam_testing_alignments_dir="data/lg_paper_data/lg_PfamTestingAlignments",
    )  # Fig. 1e, 'fast' version with FastTree instead of PhyML.
    # ~ 10 min

    print("Creating figure 1e done!")
