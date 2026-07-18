# Few-shot experiments in hmof
#for target in 'H2-77-2' \
#	'CO2-298-2.5' \
#	'CH4-298-2.5' \
#	'N2-298-0.9' \
#	'Xe-273-10' \
#	'Kr-273-10'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/hmof/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=6
#		done
#	done
	

# Few-shot experiments in hmofheat
# 500 epochs are enough for convergence
#for target in 'CO2-298K-2.5bar' 'H2-77K-2bar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/hmofheat/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/hmofheat/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/hmofheat/' \
#			--trainer.max_epochs=500 \
#			--n_frozen_layers=7
#		done
#	done


# Few-shot experiments in hcof
#for target in 'lowbar' 'highbar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/cof/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/cof/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/cof/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=6
#		done
#	done

# Few-shot experiments in ppn
#for target in '1bar' '65bar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/ppn/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/ppn/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/ppn/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=2
#		done
#	done
	
# Few-shot experiments in zeolite
#for target in 'unitless_KH' 'heat_of_adsorption'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/zeolite/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/zeolite/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/zeolite/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=1
#		done
#	done

# Few-shot experiments in c3h6c3h8coremof
#for target in 'C3H8_C3H6_Selectivity_1bar' \
#	'C3H8_C3H6_Selectivity_infinite' \
#	'C3H6_loadings' \
#	'C3H8_loadings' \
#	'TSN_S_1Bar' \
#       	'C3H8_Henry_298K' \
#	'C3H6_Henry_298K'; do
#
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/c3h6c3h8coremof/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=2
#		done
#	done
	
# Few-shot experiments in ch4n2
#for target in 'ch4n2ratio-0.1bar' \
#	'ch4n2ratio-10bar' \
#	'ch4n2ratio-1bar' \
#	'ch4uptake-0.1bar' \
#	'ch4uptake-10bar' \
#	'ch4uptake-1bar' \
#	'n2uptake-0.1bar' \
#for target in 'n2uptake-10bar' \
#	'n2uptake-1bar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 1000]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/ch4n2/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/SpbNet/benchmark/ch4n2/all.csv' \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/ch4n2/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=2
#		done
#	done

# Few-shot experiments in CoRE COFs
#for target in 'H2_log_Henry' \
#	'O2_log_Henry' \
#	'N2_log_Henry' \
#	'CO2_log_Henry' \
#	'CH4_log_Henry'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python evaluate.py -c configs/evaluate.yaml \
#			--ckpt_path=${ckpt_path} \
#			--train_sizes='[50, 100, 300, 500, 800]' \
#			--n_runs='[10, 10, 5, 5, 3]' \
#			--target=${target} \
#			--index_col='ID' \
#			--voxels_path='/home/asarikas/databases/COFSpace/voxels_data_GS32_CB30' \
#			--labels_path="/home/asarikas/databases/COFSpace/csv_data/${target}.csv" \
#			--trainer.default_root_dir='experiments/evaluate/few_shot/corecof/' \
#			--trainer.max_epochs=1000 \
#			--n_frozen_layers=2
#		done
#	done

# Few-shot experiments for H2O in UO
for target in 'log_henry_coefficient_H2O_298K [mol/kg/bar]'; do

	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"

	for ckpt_path in 'scratch' 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do

		python evaluate.py -c configs/evaluate.yaml \
			--ckpt_path=${ckpt_path} \
			--train_sizes='[50, 100, 300, 500, 1000]' \
			--n_runs='[10, 10, 5, 5, 3]' \
			--target="${target}" \
			--index_col='name' \
			--voxels_path='/home/asarikas/temp/UO_water/voxels_data' \
			--labels_path='/home/asarikas/temp/UO_water/UO_henry.csv' \
			--trainer.default_root_dir='experiments/evaluate/few_shot/uo_water/' \
			--trainer.max_epochs=1000 \
			--n_frozen_layers=2
		done
	done
