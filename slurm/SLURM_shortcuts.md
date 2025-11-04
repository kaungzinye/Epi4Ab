# Submit the SLURM job
cd /leonardo_work/AIFAC_F01_302/Epi4Ab
sbatch slurm/process_data.sbatch

# Check job status
squeue -j 21937870

# Monitor logs in real-time (once it starts)
tail -f /leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-process-21937870.out

# Check for errors
tail -f /leonardo_work/AIFAC_F01_302/Epi4Ab/logs/epi4ab-process-21937870.err