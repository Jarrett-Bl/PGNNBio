#!/bin/bash

DATABASES=("kegg" "hallmark" "wiki_pathways")
MODELS=("ann" "gnn" "pgnn" "megagnn" "kpnn")
PATHWAY_IMPORTANCE_TYPE=("naive")

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

generate_job_name() {
	local database=$1
	local model=$2

	echo "${database}_${model}.sh"
}

for database in "${DATABASES[@]}"; do
    for model in "${MODELS[@]}"; do
        for importance_type in "${PATHWAY_IMPORTANCE_TYPE[@]}"; do
            args="--database $database --model $model --pathway_importance_type $importance_type"
            job_name=$(generate_job_name "$database" "$model" "$pathway_importance_type") 

        cat <<EOT > "$TMP_DIR/$job_name"
#!/bin/bash	

#SBATCH -N 1                      # number of nodes
#SBATCH -c 32                     # number of cores 
#SBATCH -t 7-00:00:00             # time in d-hh:mm:ss
#SBATCH -p general                # partition 
#SBATCH -q public                 # QOS
#SBATCH -o artifacts/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e artifacts/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL           # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE             # Purge the job-submitting shell environment

module load mamba/latest
source activate bio_eval
python bioinformatics-sota-eval/main.py $args

EOT
        done
    done
done

for job_script in "$TMP_DIR"/*.sh; do
	sbatch "$job_script"
	sleep 1
done
