# Benchmark in hmof
#for target in 'CO2-298-2.5' \
#	'CH4-298-2.5' \
#	'N2-298-0.9' \
#	'Xe-273-10' \
#	'Kr-273-10' \
#	'H2-77-2'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/hmof/' \
#		--trainer.max_epochs=50 \
#		--n_frozen_layers=6
#	done

# Benchmark in hmofheat
#for target in 'CO2-298K-2.5bar' 'H2-77K-2bar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[5000]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/hmofheat/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/hmofheat/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/hmofheat/' \
#		--trainer.max_epochs=50 \
#		--n_frozen_layers=7
#	done

# Benchmark in hcof
#for target in 'lowbar' 'highbar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/cof/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/cof/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/cof/' \
#		--trainer.max_epochs=500 \
#		--n_frozen_layers=6
#	done

# Benchmark in ppn
#for target in '1bar' '65bar'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/ppn/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/ppn/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/ppn/' \
#		--trainer.max_epochs=100 \
#		--n_frozen_layers=2
#	done

# Benchmark in zeolite
#for target in 'unitless_KH' 'heat_of_adsorption'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/zeolite/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/zeolite/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/zeolite/' \
#		--trainer.max_epochs=500 \
#		--n_frozen_layers=1
#	done
	
# Benchmark in c3h6c3h8coremof
#for target in 'C3H8_C3H6_Selectivity_1bar' \
#	'C3H8_C3H6_Selectivity_infinite' \
#	'C3H6_loadings'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/c3h6c3h8coremof/' \
#		--trainer.max_epochs=1000 \
#		--n_frozen_layers=1
#	done
#	
## Benchmark in c3h6c3h8coremof
#for target in 'C3H8_loadings' \
#	'TSN_S_1Bar' \
#       	'C3H8_Henry_298K' \
#	'C3H6_Henry_298K'; do
#
#	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"
#
#	python evaluate.py -c configs/evaluate.yaml \
#		--ckpt_path='experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt' \
#		--train_sizes='[null]' \
#		--n_runs='[1]' \
#		--target=${target} \
#		--voxels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/voxels_data_GS32_CB30' \
#		--labels_path='/home/asarikas/databases/SpbNet/benchmark/c3h6c3h8coremof/all.csv' \
#		--trainer.default_root_dir='experiments/evaluate/benchmark/c3h6c3h8coremof/' \
#		--trainer.max_epochs=1000 \
#		--n_frozen_layers=2
#	done

for target in 'ch4n2ratio-0.1bar' \
	'ch4n2ratio-10bar' \
	'ch4n2ratio-1bar' \
	'ch4uptake-0.1bar' \
	'ch4uptake-10bar' \
	'ch4uptake-1bar' \
	'n2uptake-0.1bar' \
	'n2uptake-10bar' \
	'n2uptake-1bar'; do

	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"

	for ckpt_path in experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt; do

		python evaluate.py -c configs/evaluate.yaml \
			--ckpt_path=${ckpt_path} \
			--train_sizes='[null]' \
			--n_runs='[1]' \
			--target=${target} \
			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/ch4n2/voxels_data_GS32_CB30' \
			--labels_path='/home/asarikas/databases/SpbNet/benchmark/ch4n2/all.csv' \
			--trainer.default_root_dir='experiments/evaluate/benchmark/ch4n2/' \
			--trainer.max_epochs=1000 \
			--n_frozen_layers=2
		done
	done
