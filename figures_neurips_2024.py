import figures
from cherryml.config import create_config_from_dict
import sys
from cherryml.constants_neurips import get_configs_and_styles_from_name

num_processes = 10

def efficiency():
    simulated_data_dirs_fig_1bc = None
    labels = [
        "CherryML with FastCherries",
        "CherryML with FastTree",
        "CherryML with true trees",
    ]
    configs, styles, colors = get_configs_and_styles_from_name(labels)
    figures.fig_computational_and_stat_eff_cherry_vs_em(
        tree_estimator_config_list = configs,
        labels = labels,
        styles = styles,
        colors = colors,
        num_iterations_list = [4]*len(labels),
        add_em = False,
        simulated_data_dirs=simulated_data_dirs_fig_1bc,
        num_processes_tree_estimation=num_processes,
        supplemental_plots=False,
        num_rate_categories=4,
        output_image_dir="neurips_figures/simulated_estimated",
        normalize_learned_rate_matrix=True, 
        num_families_train_list=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        fontsize=15,
    )

def reproduce_lg():
    figures.fig_lg_paper(
        evaluation_phylogeny_estimator_name="PhyML",
        output_image_dir="neurips_figures/lg_reproduced",
        num_processes=9,
        rate_estimator_names=[
            ("reproduced WAG", "WAG\nrate \nmatrix"),
            ("reproduced LG", "LG\nrate \nmatrix"),
            ("Cherry++__4__FastTree_4rc", "LG model\n CherryML\n with \nFastTree"),
            ("Cherry++__4__FastCherries_4rc", "LG model\n CherryML\n with \nFastCherries"),
        ],
        phylogeny_estimator_configs=[
            create_config_from_dict({"identifier":"fast_tree", "args":{"num_rate_categories":1}}),
            create_config_from_dict({"identifier":"fast_tree", "args":{"num_rate_categories":1}}),
            create_config_from_dict({"identifier":"fast_tree", "args":{"num_rate_categories":4}}),
            create_config_from_dict({"identifier":"fast_cherries", 
                                    "args":{"num_rate_categories":4, "max_iters":50}}
                                    ),
        ],
        lg_pfam_training_alignments_dir="data/lg_paper_data/lg_PfamTrainingAlignments",
        lg_pfam_testing_alignments_dir="data/lg_paper_data/lg_PfamTestingAlignments",
        include_title=False,
        fontsize=15,
    )  
    print("Creating figure 1e done!")
def qmaker():
    labels = [
        "CherryML with FastTree",
        "CherryML with FastCherries",
    ]
    configs, _, _ = get_configs_and_styles_from_name(labels)

    figures.fig_qmaker(clade_name="plant", 
                       output_image_dir_prefix = "neurips_figures/fig_qmaker",
                       add_em=False, 
                       add_LG=True,
                       num_iterations=4,
                       tree_estimator_names_list=labels,
                       tree_estimator_config_list=configs,
                       num_processes_tree_estimation=num_processes,
                      fontsize=15,
    )

    figures.fig_qmaker(clade_name="insect", 
                       output_image_dir_prefix = "neurips_figures/fig_qmaker",
                       add_em=False, 
                       add_LG=True,
                       num_iterations=4,
                       tree_estimator_names_list=labels,
                       tree_estimator_config_list=configs,
                       num_processes_tree_estimation=num_processes,
                       fontsize=15,
    )

    figures.fig_qmaker(clade_name="bird", 
                       output_image_dir_prefix = "neurips_figures/fig_qmaker",
                       add_em=False, 
                       add_LG=True,
                       num_iterations=4,
                       tree_estimator_names_list=labels,
                       tree_estimator_config_list=configs,
                       num_processes_tree_estimation=num_processes,
                       fontsize=15,
    )
    figures.fig_qmaker(clade_name="mammal", 
                       output_image_dir_prefix = "neurips_figures/fig_qmaker",
                       add_em=False, 
                       add_LG=True,
                       num_iterations=4,
                       tree_estimator_names_list=labels,
                       tree_estimator_config_list=configs,
                       num_processes_tree_estimation=num_processes,
                       fontsize=15,
    )

if __name__ == "__main__":
    qmaker()
    reproduce_lg()
    efficiency()
