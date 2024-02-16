import os
import json

ds_config = {
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "offload_param": {
    "device": "cpu",
    "pin_memory": True,
  }
  },
  "fp16": {
    "enabled": False,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_max_lr": "auto",
      "warmup_num_steps": 'auto',
      "total_num_steps": 'auto',
    }
  },
  "optimizer": {
    "type": "AdamW", 
    "params": {
    "lr": "auto", 
    "betas": [0.9, 0.999], 
    "eps": 1e-08, "weight_decay": 0.0
    }
    },
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 1,
  "wall_clock_breakdown": False
}


with open(os.path.join('ds_zero3_cpu_offload.json'), 'w') as f:
    json.dump(ds_config, f)