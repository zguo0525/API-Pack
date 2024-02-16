export HF_DATASETS_CACHE=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cache
export OMP_NUM_THREADS=10
export LD_LIBRARY_PATH=/gpfs/u/home/SIFA/SIFAzhnu/scratch/cudnn-11.3-linux-ppc64le-v8.2.1.32/targets/ppc64le-linux/lib:$LD_LIBRARY_PATH

set -e
set -x

# Define an array of languages
langs=("curl" "go" "java" "javascript" "libcurl" "node" "php" "python" "ruby" "swift")
#langs=("php" "python" "ruby" "swift")

for lang in "${langs[@]}"; do
        export EXP_NAME="${lang}"
        echo $EXP_NAME

        COMMAND="python data_aug.py ${lang}"

        sbatch inference.sh "${EXP_NAME}" "$COMMAND"

        echo "Sleeping for 2 seconds..."
        sleep 1
    done
done