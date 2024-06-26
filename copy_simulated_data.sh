mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/msa_dir && time cp -r _cache_benchmarking/subset_msa_to_leaf_nodes/b8523c83b3e5478556b367a5f5bf5428e9f3244ce206f180f40854b716a4a6bd/output_msa_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/msa_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_tree_dir && time cp -r _cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_tree_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_tree_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_site_rates_dir && time cp -r _cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_site_rates_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_site_rates_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_likelihood_dir && time cp -r _cache_benchmarking/fast_tree/6fe66dc295954e80b8298ea323c4574995999b38653c0bc2c4c08161c502d2b4/output_likelihood_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1d/gt_likelihood_dir/



mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/msa_dir && time cp -r _cache_benchmarking/subset_msa_to_leaf_nodes/40aaa15b51d7ca7f9fe0d55550947207f53bdc7def3d4a0401cd905d486604db/output_msa_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/msa_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/contact_map_dir && time cp -r _cache_benchmarking/create_maximal_matching_contact_map/492b760db9307f08efecec9ba203d84cdccecbe80ec8007d25580d595b342f7c/o_contact_map_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/contact_map_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_tree_dir && time cp -r _cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_tree_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_tree_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_site_rates_dir && time cp -r _cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_site_rates_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_site_rates_dir/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_likelihood_dir && time cp -r _cache_benchmarking/fast_tree/37c5828031a53e44ec03f5ace1d09ec4d959b41cd80a2479f17e2328dc9f923b/output_likelihood_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_2ab/gt_likelihood_dir/



mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/msa_dir && time cp -r _cache_benchmarking_em/subset_msa_to_leaf_nodes/da6ef3b0d58b16b6b9f5ad5628ffe750fc41b6b444a12b32f4f5d904a220facf/output_msa_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/msa_dir/  # 7m49s

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_tree_dir && time cp -r _cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_tree_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_tree_dir/  # 0m33s

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_site_rates_dir && time cp -r _cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_site_rates_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_site_rates_dir/  # 0m25s

mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_likelihood_dir && time cp -r _cache_benchmarking_em/fast_tree/eecc0e2b9e570733bd4817b6bc57abd38ef31652f30cc17612d66147203398a2/output_likelihood_dir/*.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/gt_likelihood_dir/  # 0m50s

time cp -r fig_1bc_simulated_data_families_all.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/fig_1bc/  # 0s



mkdir /global/scratch/users/sprillo/cherryml_simulated_datasets/rate_matrices

cp _cache_benchmarking/quantized_transitions_mle/9858a0319e46041903de4c8b87147b96549e7b8991086574025f2a792a1621c9/output_rate_matrix_dir/result.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/rate_matrices/Q2.txt

cp data/rate_matrices/lg.txt /global/scratch/users/sprillo/cherryml_simulated_datasets/rate_matrices/lg.txt


pushd /global/scratch/users/sprillo/cherryml_simulated_datasets/

tar -cvzf fig_1d.tgz fig_1d/

tar -cvzf fig_2ab.tgz fig_2ab/

tar -cvzf fig_1bc.tgz fig_1bc/

tar -cvzf rate_matrices.tgz rate_matrices/

popd
