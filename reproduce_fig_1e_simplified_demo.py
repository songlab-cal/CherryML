import figures

from cherryml.config import create_config_from_dict

if __name__ == "__main__":
    print("Creating figure 1e with FastTree instead of PhyML ...")

    figures.fig_lg_paper(
        evaluation_phylogeny_estimator_name="FastTree",
        output_image_dir="fig_1e_simplified/",
        num_processes=32,
        rate_estimator_names=[
            ("reproduced WAG", "WAG\nrate matrix"),
            ("reproduced LG", "LG\nrate matrix"),
            ("Cherry++__4__FastTree_4rc", "LG model\n CherryML\n with \nFastTree 4rc"),
        ],
        phylogeny_estimator_configs=[
            create_config_from_dict({"identifier":"", "args":{}}),
            create_config_from_dict({"identifier":"", "args":{}}),
            create_config_from_dict({"identifier":"fast_tree", 
                                    "args":{"num_rate_categories":4}}
                                    ),
        ],
        lg_pfam_training_alignments_dir="data/lg_paper_data/lg_PfamTrainingAlignments",
        lg_pfam_testing_alignments_dir="data/lg_paper_data/lg_PfamTestingAlignments",
    )  # Fig. 1e, 'fast' version with FastTree instead of PhyML.
    # ~ 10 min

    print("Creating figure 1e done!")
