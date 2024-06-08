import sys
import logging
import datasets
import transformers
import accelerate

formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(filename)s - %(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)

    accelerator = accelerate.Accelerator()
    if accelerator.is_main_process:
        logger.setLevel(logging.INFO)
        # datasets.utils.logging.set_verbosity_info()
        # transformers.utils.logging.set_verbosity_info()
    else:
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return logger