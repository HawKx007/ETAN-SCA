# src/tb_logger.py
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

def get_tb_writer(run_name, base_dir="runs"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    return SummaryWriter(log_dir=str(log_dir))

