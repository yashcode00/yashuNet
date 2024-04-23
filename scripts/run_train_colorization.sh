#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --partition=nltmp
#SBATCH --job-name=colr-yamaha
#SBATCH --gres=gpu:2
#SBATCH --output=/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/logs/out.log  # Updated output path
#SBATCH --error=/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/logs/err.log    # Updated error path
#SBATCH --time=7-0:0:0  # 7 days, 0 hours, 0 minutes, and 0 seconds (you can adjust this as needed)

# Define the Conda environment, activate it, and define the Python script and log file
log_dir="/nlsasfs/home/nltm-st/sujitk/temp/yashuNet/logs/"
output_main="${log_dir}train-color.log"

eval "$(conda shell.bash hook)" &> /nlsasfs/home/nltm-st/sujitk/temp/yashuNet/logs/error.txt

# Activate the Conda environment
conda activate yamaha

# Set proxy environment variables 
export HTTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export HTTPS_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export FTP_PROXY='http://proxy-10g.10g.siddhi.param:9090'
export ALL_PROXY='http://proxy-10g.10g.siddhi.param:9090'

# Run Python script in the background and save the output to the log file
# accelerate launch /nlsasfs/home/nltm-st/sujitk/yash-mtp/src/tdnn/xVectorTraining-ddp.py &> "$output_main" &
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 12335 /nlsasfs/home/nltm-st/sujitk/temp/yashuNet/src/train_colorization.py &> "$output_main" &

# python3 /nlsasfs/home/nltm-st/sujitk/temp/eeg2text/src/common/utils/data.py  &> "$output_main" &

# Save the background job's process ID (PID)
bg_pid=$!

# Print a message indicating that the job is running in the background
echo "Job is running in the background with PID $bg_pid."

# Deactivate the Conda environment (optional)
# conda deactivate  

# Wait for the background job to complete
wait $bg_pid

# Deactivate the Conda environment (if not done already)
conda deactivate