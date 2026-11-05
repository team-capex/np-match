import threading
import time

import torch.cuda


def monitor_gpu(
    interval: float = 10.0,
    device: str = "cuda",
    name: str = "0",
):
    if device == "cpu" or not torch.cuda.is_available():
        return

    def print_utilization():
        torch.cuda.reset_peak_memory_stats(device)
        gb = 1024**3
        while True:
            utilization = torch.cuda.utilization(device)
            _, mem_total = torch.cuda.mem_get_info(device)
            mem_alloc = torch.cuda.memory_allocated(device)
            max_mem_alloc = torch.cuda.max_memory_allocated(device)
            print(
                f"Device {name} - utilization: {utilization}%, "
                f"memory: {mem_alloc / gb:.1f}/{mem_total / gb:.1f} GB, "
                f"max memory: {max_mem_alloc / gb:.1f}/{mem_total / gb:.1f} GB",
                flush=True,
            )
            time.sleep(interval)

    threading.Thread(target=print_utilization, daemon=True).start()
