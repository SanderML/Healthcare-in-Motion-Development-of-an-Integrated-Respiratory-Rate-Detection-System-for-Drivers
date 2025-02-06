default:
    @just --list

build_sing:
    cd container; singularity build --fakeroot singularity.sif singularity.def
    
check_package_versions:
    singularity exec --nv container/singularity.sif python -m pip freeze

check_gpu:
    singularity exec --nv container/singularity.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

preprocessing_subjects:
    singularity exec --nv container/singularity.sif python src/preprocessing/preprocessing_subjects.py

main *ARGS:
    singularity exec --nv container/singularity.sif python main.py {{ ARGS }}

run_jobs *ARGS:
    singularity exec --nv container/singularity.sif python src/utils/run_jobs.py {{ ARGS }}