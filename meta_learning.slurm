#!/usr/bin/env bash
#SBATCH --job-name=cnn_attn5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=18:00:00       # Target: ~6 h instead of 12 h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export SDL_AUDIODRIVER=dummy
export AUDIODEV=null

echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# ── Thread-count pins (one thread per Ray worker is optimal; GPU gets all cores) ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Ray: avoid /tmp filling up on shared clusters ──
export RAY_TMPDIR="${HOME}/tmp/ray"
mkdir -p "${RAY_TMPDIR}"

export TEACHER_MODES_DEFAULT='[
  ["expert", {"expert_teacher": true, "unrealiable_teacher": false, "eps": 0.10}],
  ["learning",   {"expert_teacher": false, "unrealiable_teacher": false,
              "alpha": [0.0, 0.288, 0.288], "tau": [0.33, 0.33, 0.33]}],
  ["unreliable", {"expert_teacher": false, "unrealiable_teacher": true}]
]'

#export TEACHER_MODES_DEFAULT='[
#  ["learning",   {"expert_teacher": false, "unrealiable_teacher": false,
#                "alpha": [0.0, 0.288, 0.288], "tau": [0.33, 0.33, 0.33]}]
#  ]'

#export TEACHER_MODES_DEFAULT='[
#    ["unreliable", {"expert_teacher": false, "unrealiable_teacher": true}]
#  ]'
  


python bandit_train_batch_evalgrid_ppo_gridK_aux_postgrid_v4_gtrace_npz.py \
  --grid-repeats 16              \
  --num-updates 1500		 \
  --warmup-updates 500           \
  --num-workers 39               \
  --ppo-epochs 2                 \
  --train-session-chunk-size 16  \
  --eval-interval 50             \
  --eval-sessions-per-cell 20    \
  --patience-evals 10		 \
  --train-memory-lambda 1.0      \
#  --ablation-g 0.2         

  # --no-gate-ol-sg   # uncomment to disable stop-gradient on g_ol (default: sg ON, matching ↓ formula)

# ── Notes on tuned flags vs original ──
#
#  --train-session-chunk-size 80   (was 70)
#    The pipelined rollout+update lets you use the full batch (B=80) as one
#    chunk, halving the number of forward/backward passes per PPO epoch from 2→1.
#    Memory impact is small because observations are now uint8 on CPU.
#
#  All other flags kept identical so results are directly comparable.

#python bandit_train_batch_evalgrid_ppo_gridK_aux_postgrid_v4_gtrace_npz.py --grid-repeats 1 --num-workers 40 --ppo-epochs 1 --ppo-minibatch-size 64   --eval-interval 20  --eval-sessions-per-cell 1  --num-updates 5  --warmup-updates 1  --patience-evals 0 --eval-interval 1
