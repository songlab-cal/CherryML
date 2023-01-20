import figures

from cherryml.markov_chain import get_jtt_path, get_wag_path, get_lg_path

from cherryml import caching
# caching.set_log_level(9)
# caching.set_read_only(True)


if __name__ == "__main__":
    print("Creating figures ...")

    # figures.fig_single_site_quantization_error()  # Fig. 1d
    # # ~ 8hr

    # figures.fig_pair_site_quantization_error(
    #     Q_2_name="unmasked-co-transitions",
    # )  # Fig. 2a
    # # ~ 9hr

    # figures.fig_pair_site_quantization_error(
    #     Q_2_name="unmasked-single-transitions",
    # )  # Fig. 2b
    # # ~ 30 min

    # figures.fig_site_rates_vs_number_of_contacts()  # Fig. 2e
    # # ~ 9 hr

    # figures.fig_coevolution_vs_indep()  # Fig. 2c, 2d
    # # ~ 1 min

    # figures.fig_MSA_VI_cotransition()  # Comment in paragraph.
    # # ~ 1 min

    # figures.fig_lg_paper(num_processes=8)  # Fig. 1e
    # ~ 20 hr

    extra_em_command_line_args = "-log 6 -f 3 -mi 0.000001"
    # extra_em_command_line_args = "-log 6 -f 3 -mi 0.1"
    num_processes_tree_estimation = 4
    # for include_qmaker_matrices in [False, True]:
    for include_qmaker_matrices in [False]:
        figures.fig_qmaker(num_processes_tree_estimation=num_processes_tree_estimation, extra_em_command_line_args=extra_em_command_line_args, include_qmaker_matrices=include_qmaker_matrices, clade_name="plant")
        figures.fig_qmaker(num_processes_tree_estimation=num_processes_tree_estimation, extra_em_command_line_args=extra_em_command_line_args, include_qmaker_matrices=include_qmaker_matrices, clade_name="insect")
        figures.fig_qmaker(num_processes_tree_estimation=num_processes_tree_estimation, extra_em_command_line_args=extra_em_command_line_args, include_qmaker_matrices=include_qmaker_matrices, clade_name="bird")
        figures.fig_qmaker(num_processes_tree_estimation=num_processes_tree_estimation, extra_em_command_line_args=extra_em_command_line_args, include_qmaker_matrices=include_qmaker_matrices, clade_name="mammal")
        # figures.fig_qmaker__revision(num_processes_tree_estimation=num_processes_tree_estimation, extra_em_command_line_args=extra_em_command_line_args, include_qmaker_matrices=include_qmaker_matrices, clade_name="yeast", add_em=False)  # EM too slow...


    # # Current manuscript
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     em_backend="historian",
    #     em_init="jtt-ipw",
    # )  # Fig. 1b, 1c
    # # More iterations in Historian
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.0000001 -maxiter 100000000 -nolaplace",
    #     em_backend="historian",
    #     em_init="jtt-ipw",
    # )  # Fig. 1b, 1c
    # XRATE, too few iterations
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.00001",
    #     em_backend="xrate",
    #     em_init="jtt-ipw",
    # )  # Fig. 1b, 1c
    # # XRATE, OK iterations
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #     em_backend="xrate",
    #     em_init="jtt-ipw",
    # )  # Fig. 1b, 1c
    # # XRATE, more iterations (shows that it has converged)
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.0000001",
    #     em_backend="xrate",
    #     em_init="jtt-ipw",
    # )  # Fig. 1b, 1c

    # # TODO: See if initializing with CherryML makes XRATE faster
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #     em_backend="xrate",
    #     em_init="cherryml",
    # )  # Fig. 1b, 1c
    # # TODO: See if initializing with EQU makes XRATE slower
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #     em_backend="xrate",
    #     em_init="equ",
    # )  # Fig. 1b, 1c

    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     em_backend="historian",
    # )  # Fig. 1b, 1c
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.0000001 -maxiter 100000000 -nolaplace",
    #     em_backend="historian",
    # )  # Fig. 1b, 1c
    # caching.set_read_only(True)
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # # ~ 62 hr
    # # caching.set_read_only(True)
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.001",
    #     em_backend="xrate",
    #     em_init="equ",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     # extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     # em_backend="historian",
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.0001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     # extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     # em_backend="historian",
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.00001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     # extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     # em_backend="historian",
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     # extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     # em_backend="historian",
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.0000001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     # extra_em_command_line_args="-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     # em_backend="historian",
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.00000001",
    #     em_backend="xrate",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr

    # With XRATE, try CherryML initialization to show that we can speed up EM! (n0166)
    # caching.set_read_only(True)  # HERE
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.001",
    #     em_backend="xrate",
    #     em_init="cherryml",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr
    # # With XRATE, try CherryML initialization to show that we can speed up EM! (n0166)
    # figures.fig_computational_and_stat_eff_cherry_vs_em(
    #     extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #     em_backend="xrate",
    #     em_init="cherryml",
    # )  # Fig. 1b, 1c
    # # ~ 62 hr

    # figures.fig_relearn_LG_on_pfam15k()
    # figures.fig_relearn_LG_on_pfam15k_vary_num_families_train()

    # # Show that we need few families to outperform LG.
    # for num_sequences in [16, 128, 1024]:
    #     figures.fig_relearn_LG_on_pfam15k_vary_num_families_train(
    #         num_sequences_train=num_sequences,
    #         num_sequences_test=num_sequences,
    #         num_families_train_list=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12000],
    #     )

    # # Show that EM also does better on 16 train for 16 test than 32 train
    # num_sequences_test = 16
    # for use_simulated_data in [True, False]:
    #     figures.fig_relearn_LG_on_pfam15k_vary_num_families_train__em(
    #         num_sequences_train=16,
    #         num_sequences_test=num_sequences_test,
    #         num_families_train_list=[32, 64, 128, 256, 512, 1024, 2048],  #, 4096, 8192, 12000],
    #         extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #         em_backend="xrate",
    #         use_simulated_data=use_simulated_data,
    #     )
    #     figures.fig_relearn_LG_on_pfam15k_vary_num_families_train__em(
    #         num_sequences_train=32,
    #         num_sequences_test=num_sequences_test,
    #         num_families_train_list=[32, 64, 128, 256, 512, 1024], #, 2048, 4096, 8192, 12000],
    #         extra_em_command_line_args="-log 6 -f 3 -mi 0.000001",
    #         em_backend="xrate",
    #         use_simulated_data=use_simulated_data,
    #     )
    #     figures.fig_relearn_LG_on_pfam15k_vary_num_families_train(
    #         num_sequences_train=16,
    #         num_sequences_test=num_sequences_test,
    #         num_families_train_list=[32, 64, 128, 256, 512, 1024, 2048],  #, 4096, 8192, 12000],
    #         use_simulated_data=use_simulated_data,
    #     )
    #     figures.fig_relearn_LG_on_pfam15k_vary_num_families_train(
    #         num_sequences_train=32,
    #         num_sequences_test=num_sequences_test,
    #         num_families_train_list=[32, 64, 128, 256, 512, 1024], #, 2048, 4096, 8192, 12000],
    #         use_simulated_data=use_simulated_data,
    #     )

    # # num_sequences_test X num_sequences_train phenomenon!
    # for use_simulated_data in [False, True]:
    #     for num_sequences_train in [16, 128, 1024]:
    #         for num_sequences_test in [16, 128, 1024]:
    #             figures.fig_relearn_LG_on_pfam15k_vary_num_families_train(
    #                 num_sequences_train=num_sequences_train,
    #                 num_sequences_test=num_sequences_test,
    #                 num_families_train_list=[12000],
    #                 use_simulated_data=use_simulated_data,
    #             )

    # # ***** QMaker candidate Figures (in particular 3rd row I showed to Yuns) ****
    # # DONE: CherryML does not do too bad. It is best with 2 iterations. Interestingly, on insect outcompetes QMaker a lot.
    # for (evaluator_name, extra_evaluator_command_line_args) in [("FastTree", "")]: # , ("ModelFinder", "")]:
    #     # num_families_test = None
    #     # for num_iterations in [2]:
    #     #     for add_cherryml in [False, True]:
    #     #         figures.fig_qmaker_pfam(
    #     #             num_families_test=num_families_test,
    #     #             add_cherryml=add_cherryml,
    #     #             num_iterations=num_iterations,
    #     #             initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #     #             evaluator_name=evaluator_name,
    #     #             extra_evaluator_command_line_args=extra_evaluator_command_line_args,
    #     #         )
    #     #         for clade_name in ["plant", "bird", "mammal", "insect", "yeast"]:
    #     #             figures.fig_qmaker_clade(
    #     #                 clade_name=clade_name,
    #     #                 num_families_test=num_families_test,
    #     #                 add_cherryml=add_cherryml,
    #     #                 num_iterations=num_iterations,
    #     #                 initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #     #                 evaluator_name=evaluator_name,
    #     #                 extra_evaluator_command_line_args=extra_evaluator_command_line_args,
    #     #             )
    #     # DONE: See how adding EM looks like!
    #     num_families_test = None
    #     for extra_em_command_line_args in ["-log 6 -f 3 -mi 0.000001", "-log 6 -f 3 -mi 0.1"]:
    #         for num_iterations in [2]:
    #             for add_cherryml_em in [True]:
    #                 # figures.fig_qmaker_pfam(  # Too slow for XRATE!
    #                 #     num_families_test=num_families_test,
    #                 #     add_cherryml=add_cherryml_em,
    #                 #     add_em=add_cherryml_em,
    #                 #     num_iterations=num_iterations,
    #                 #     initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #                 #     evaluator_name=evaluator_name,
    #                 #     extra_evaluator_command_line_args=extra_evaluator_command_line_args,
    #                 # )
    #                 for clade_name in ["plant", "insect", "bird", "mammal"]:
    #                     figures.fig_qmaker_clade(
    #                         clade_name=clade_name,
    #                         num_families_test=num_families_test,
    #                         add_cherryml=add_cherryml_em,
    #                         add_em=add_cherryml_em,
    #                         extra_em_command_line_args=extra_em_command_line_args,
    #                         num_iterations=num_iterations,
    #                         initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #                         evaluator_name=evaluator_name,
    #                         extra_evaluator_command_line_args=extra_evaluator_command_line_args,
    #                     )
    #     # # TODO: yeast still hasn't finished...
    #     # num_families_test = None
    #     # for num_iterations in [2]:
    #     #     for add_cherryml_em in [True]:
    #     #         # figures.fig_qmaker_pfam(  # Too slow for XRATE!
    #     #         #     num_families_test=num_families_test,
    #     #         #     add_cherryml=add_cherryml_em,
    #     #         #     add_em=add_cherryml_em,
    #     #         #     num_iterations=num_iterations,
    #     #         #     initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #     #         # )
    #     #         for clade_name in ["yeast"]: # TODO: See how new clades do for EM!
    #     #             figures.fig_qmaker_clade(
    #     #                 clade_name=clade_name,
    #     #                 num_families_test=num_families_test,
    #     #                 add_cherryml=add_cherryml_em,
    #     #                 add_em=add_cherryml_em,
    #     #                 num_iterations=num_iterations,
    #     #                 initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #     #             )

    # # New stuff after IQTree integration and PhyML
    # # DONE: IQTree performs worse than FastTree! In a sense this is good because it shows that tree estimation DOES indeed affect performance! Because of this, I Will not try out ModelFinder as the tree estimator. If it does not work well, I will implement my own version with FastTree!
    # num_families_test = None
    # for num_iterations in [2]:
    #     for add_cherryml in [False, True]:
    #         for clade_name in ["plant"]:
    #             figures.fig_qmaker_clade(
    #                 clade_name=clade_name,
    #                 num_families_test=num_families_test,
    #                 add_cherryml=add_cherryml,
    #                 num_iterations=num_iterations,
                    
    #                 # tree_estimator_name="FastTree",
    #                 # extra_tree_estimator_command_line_args="",
    #                 # tree_estimator_name="IQTree-G-posterior_mean",
    #                 # extra_tree_estimator_command_line_args="-fast",
    #                 # tree_estimator_name="IQTree-G-MAP",
    #                 # extra_tree_estimator_command_line_args="-fast",
    #                 # tree_estimator_name="IQTree-G-MAP",  # 1hr
    #                 # extra_tree_estimator_command_line_args="",
    #                 # tree_estimator_name="IQTree-G-posterior_mean",  # 1hr
    #                 # extra_tree_estimator_command_line_args="",
    #                 # tree_estimator_name="IQTree-R-posterior_mean",
    #                 # extra_tree_estimator_command_line_args="-fast",
    #                 # tree_estimator_name="IQTree-R-MAP",
    #                 # extra_tree_estimator_command_line_args="-fast",
    #                 # tree_estimator_name="IQTree-R-MAP",  # 1hr
    #                 # extra_tree_estimator_command_line_args="",
    #                 # tree_estimator_name="IQTree-R-posterior_mean",  # 1hr
    #                 # extra_tree_estimator_command_line_args="",
    #                 tree_estimator_name="PhyML",  # TODO: Check results! (ETA: 24h)
    #                 extra_tree_estimator_command_line_args="",

    #                 # evaluator_name="FastTree",
    #                 # extra_evaluator_command_line_args="",
    #                 # evaluator_name="IQTree-G",
    #                 # extra_evaluator_command_line_args="-fast",
    #                 # evaluator_name="IQTree-R",
    #                 # extra_evaluator_command_line_args="-fast",
    #                 # evaluator_name="ModelFinder",
    #                 # extra_evaluator_command_line_args="",
    #                 evaluator_name="ModelFinderPlus",
    #                 extra_evaluator_command_line_args="",

    #                 initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #             )


    # TODO: Check results! My thought is we should abandon plant, etc. because the difference might come from the edge-linked partition model, which gives QMaker much better trees than we have.
    ##### Try ModelFinder tree estimator too see if it at least is better than IQTree! Hopefully also better than FastTree, but who knows. If it is not better than FastTree, I will proceed to implement ModelFinder with FastTree!
    # # Normal ModelFinder
    # num_families_test = None
    # for num_iterations in [2]:
    #     for add_cherryml in [False, True]:
    #         for clade_name in ["plant"]:
    #             figures.fig_qmaker_clade(
    #                 clade_name=clade_name,
    #                 num_families_test=num_families_test,
    #                 add_cherryml=add_cherryml,
    #                 num_iterations=num_iterations,
                    
    #                 tree_estimator_name="ModelFinderPlus-posterior_mean",
    #                 extra_tree_estimator_command_line_args="",

    #                 evaluator_name="ModelFinderPlus",
    #                 extra_evaluator_command_line_args="",

    #                 initial_tree_estimator_rate_matrix_path=get_jtt_path() + "," + get_wag_path() + "," + get_lg_path(),
    #             )

    # # TODO: Redo after MFP incorporation
    # # Try to reproduce QMaker plant as closely as possible
    # # This is how we run ModelFinder on all rate matrices at the same time (as in the paper)
    # figures.fig_qmaker_plant_deprecated()
    # # This is if we run ModelFinder on each family independently.
    # figures.fig_qmaker_clade(
    #     clade_name="plant",
    #     evaluator_name="ModelFinder",
    # )
    # # TODO: Redo after MFP incorporation
    # # Try to reproduce QMaker plant as closely as possible
    # # This is how we run ModelFinder on all rate matrices at the same time (as in the paper)
    # figures.fig_qmaker_plant()
    # # This is if we run ModelFinder on each family independently.
    # figures.fig_qmaker_clade(
    #     clade_name="plant",
    #     evaluator_name="ModelFinderPlus",
    # )

    # # TODO: Run & check results! Specifically, check what happens when EM is run with 0.01
    # # Let's see on the LG paper's figure how using different phylogeny estimators changes things.
    # # for evaluation_phylogeny_estimator_name in ["FastTree", "IQTree__G__-fast", "IQTree__R__-fast", "MF__-fast"]:
    # for evaluation_phylogeny_estimator_name in ["FastTree"]:
    #     print(f"***** evaluation_phylogeny_estimator_name = {evaluation_phylogeny_estimator_name} *****")
    #     figures.fig_lg_paper(  # TODO: Check out.
    #         evaluation_phylogeny_estimator_name=evaluation_phylogeny_estimator_name,

    #         num_processes=32,
    #         rate_estimator_names=[
    #             ("reproduced WAG", "WAG"),
    #             ("reproduced LG", "LG"),
    #             # Paper (FastTree)
    #             ("Cherry__1", "LG w/CherryML (1)"),
    #             ("Cherry__2", "LG w/CherryML (2)"),
    #             ("Cherry__3", "LG w/CherryML (3)"),
    #             ("Cherry__4", "LG w/CherryML (4)"),

    #             # Q.LG
    #             ("Q.LG", "Q.LG"),  # Estimated on LG paper's Pfam
    #             # Q.pfam
    #             ("Q.pfam", "Q.pfam"),  # Was estimated on a newer, larger version of Pfam than Q.LG

    #             # # TODO: If ModelFinderPlus is not good, try out the following which include FastTree++ (FT with multiple RMs)
    #             # #Repro after enabling multiple RMs
    #             # ("Cherry_repro_2023_01_14__1", "repro LG; w/CherryML (1)"),
    #             # ("Cherry_repro_2023_01_14__2", "repro LG; w/CherryML (2)"),
    #             # ("Cherry_repro_2023_01_14__3", "repro LG; w/CherryML (3)"),
    #             # ("Cherry_repro_2023_01_14__4", "repro LG; w/CherryML (4)"),
    #             # # See what happens if in Cherry paper we start from LG instead of EQU
    #             # ("Cherry_from_lg__1", "from lg; LG w/CherryML (1)"),
    #             # ("Cherry_from_lg__2", "from lg; LG w/CherryML (2)"),
    #             # ("Cherry_from_lg__3", "from lg; LG w/CherryML (3)"),
    #             # ("Cherry_from_lg__4", "from lg; LG w/CherryML (4)"),
    #             # # See what happens if in Cherry paper we start from JTT,WAG,LG instead of EQU, and do the QMaker concatenation trick
    #             # ("Cherry_FT++jtt_wag_lg__1", "LG++ w/CherryML (1)"),
    #             # ("Cherry_FT++jtt_wag_lg__2", "LG++ w/CherryML (2)"),
    #             # ("Cherry_FT++jtt_wag_lg__3", "LG++ w/CherryML (3)"),
    #             # ("Cherry_FT++jtt_wag_lg__4", "LG++ w/CherryML (4)"),

    #             # # IQTree
    #             # # IQTree gamma, fast, best mode (posterior_mean)
    #             # ("Cherry_IQ__1__G__posterior_mean__-fast", "IQ_G4pmf + CherryML (1)"),
    #             # ("Cherry_IQ__2__G__posterior_mean__-fast", "IQ_G4pmf + CherryML (2)"),
    #             # ("Cherry_IQ__3__G__posterior_mean__-fast", "IQ_G4pmf + CherryML (3)"),
    #             # ("Cherry_IQ__4__G__posterior_mean__-fast", "IQ_G4pmf + CherryML (4)"),
    #             # # IQTree gamma, fast, worse mode (MAP)
    #             # ("Cherry_IQ__1__G__MAP__-fast", "IQ_G4mapf + CherryML (1)"),
    #             # ("Cherry_IQ__2__G__MAP__-fast", "IQ_G4mapf + CherryML (2)"),
    #             # ("Cherry_IQ__3__G__MAP__-fast", "IQ_G4mapf + CherryML (3)"),
    #             # ("Cherry_IQ__4__G__MAP__-fast", "IQ_G4mapf + CherryML (4)"),
    #             # # IQTree gamma, normal, best mode (posterior_mean)
    #             # ("Cherry_IQ__1__G__posterior_mean__", "IQ_G4pm + CherryML (1)"),
    #             # ("Cherry_IQ__2__G__posterior_mean__", "IQ_G4pm + CherryML (2)"),
    #             # ("Cherry_IQ__3__G__posterior_mean__", "IQ_G4pm + CherryML (3)"),
    #             # ("Cherry_IQ__4__G__posterior_mean__", "IQ_G4pm + CherryML (4)"),

    #             # # Same as above (IQTree) but for PDF model
    #             # # IQTree PDF, fast, best mode (posterior_mean)
    #             # ("Cherry_IQ__1__R__posterior_mean__-fast", "IQ_R4pmf + CherryML (1)"),
    #             # ("Cherry_IQ__2__R__posterior_mean__-fast", "IQ_R4pmf + CherryML (2)"),
    #             # ("Cherry_IQ__3__R__posterior_mean__-fast", "IQ_R4pmf + CherryML (3)"),
    #             # ("Cherry_IQ__4__R__posterior_mean__-fast", "IQ_R4pmf + CherryML (4)"),                
    #             # # IQTree PDF, fast, worse mode (MAP)
    #             # ("Cherry_IQ__1__R__MAP__-fast", "IQ_R4mapf + CherryML (1)"),
    #             # ("Cherry_IQ__2__R__MAP__-fast", "IQ_R4mapf + CherryML (2)"),
    #             # ("Cherry_IQ__3__R__MAP__-fast", "IQ_R4mapf + CherryML (3)"),
    #             # ("Cherry_IQ__4__R__MAP__-fast", "IQ_R4mapf + CherryML (4)"),
    #             # # IQTree PDF, normal, best mode (posterior_mean)
    #             # ("Cherry_IQ__1__R__posterior_mean__", "IQ_R4pm + CherryML (1)"),
    #             # ("Cherry_IQ__2__R__posterior_mean__", "IQ_R4pm + CherryML (2)"),
    #             # ("Cherry_IQ__3__R__posterior_mean__", "IQ_R4pm + CherryML (3)"),
    #             # ("Cherry_IQ__4__R__posterior_mean__", "IQ_R4pm + CherryML (4)"),

    #             # # MF, fast
    #             # ("Cherry_MF__1__posterior_mean", "MFpm + w/CherryML (1)"),
    #             # ("Cherry_MF__2__posterior_mean", "MFpm + w/CherryML (2)"),
    #             # ("Cherry_MF__3__posterior_mean", "MFpm + w/CherryML (3)"),
    #             # ("Cherry_MF__4__posterior_mean", "MFpm + w/CherryML (4)"),

    #             # # MF, normal  # TODO: Hasn't finished (comment out)
    #             # ("Cherry_MFP__1__posterior_mean", "MFPpm + w/CherryML (1)"),
    #             # ("Cherry_MFP__2__posterior_mean", "MFPpm + w/CherryML (2)"),
    #             # ("Cherry_MFP__3__posterior_mean", "MFPpm + w/CherryML (3)"),
    #             # ("Cherry_MFP__4__posterior_mean", "MFPpm + w/CherryML (4)"),

    #             # # PhyML  # Takes ~ 150 minutes each iteration!
    #             # ("Cherry_P__1", "PhyML + w/CherryML (1)"),
    #             # ("Cherry_P__2", "PhyML + w/CherryML (2)"),
    #             # ("Cherry_P__3", "PhyML + w/CherryML (3)"),
    #             # ("Cherry_P__4", "PhyML + w/CherryML (4)"),

    #             # EM
    #             ("EM_FT__1__0.000001", "LG w/EM 6 (1)"),
    #             ("EM_FT__2__0.000001", "LG w/EM 6 (2)"),
    #             ("EM_FT__3__0.000001", "LG w/EM 6 (3)"),
    #             ("EM_FT__4__0.000001", "LG w/EM 6 (4)"),

    #             ("EM_FT__1__0.1", "LG w/EM 1 (1)"),
    #             ("EM_FT__2__0.1", "LG w/EM 1 (2)"),
    #             ("EM_FT__3__0.1", "LG w/EM 1 (3)"),
    #             ("EM_FT__4__0.1", "LG w/EM 1 (4)"),
    #         ],
    #     )  # Fig. 1e
        # ~ 4 hr

    # # Old evaluation protocol: FastTree
    # num_families_test = None
    # for num_iterations in [2]:
    #     for add_cherryml in [False, True]:
    #         for clade_name in ["plant"]:
    #             figures.fig_qmaker_clade(
    #                 clade_name=clade_name,
    #                 num_families_test=num_families_test,
    #                 add_cherryml=add_cherryml,
    #                 num_iterations=num_iterations,

    #                 tree_estimator_name="FastTree",
    #                 extra_tree_estimator_command_line_args="",

    #                 evaluator_name="FastTree",
    #                 extra_evaluator_command_line_args="",

    #                 initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #             )

    # # New evaluation protocol: MF, but SEPARATELY. Show that does not reproduce results.
    # num_families_test = None
    # for num_iterations in [2]:
    #     for add_cherryml in [False, True]:
    #         for clade_name in ["plant"]:
    #             figures.fig_qmaker_clade(
    #                 clade_name=clade_name,
    #                 num_families_test=num_families_test,
    #                 add_cherryml=add_cherryml,
    #                 num_iterations=num_iterations,

    #                 tree_estimator_name="FastTree",
    #                 extra_tree_estimator_command_line_args="",

    #                 evaluator_name="ModelFinder",
    #                 extra_evaluator_command_line_args="",

    #                 initial_tree_estimator_rate_matrix_path=get_lg_path(),
    #             )

    # figures.fig_qmaker_hardcoded("plant")

    print("Creating figures done!")
# 321
