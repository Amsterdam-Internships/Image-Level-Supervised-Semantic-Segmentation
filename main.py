"""
Main file that calls functions and loads configuration
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'util')

import timeit
from config import Config

# Set individual parts of implementation
CREATE_SALIENCY = False
TRAIN_SEED = False
CREATE_ATTENTION = False
CREATE_GUIDE = False
TRAIN_DEEPLAB = False
EVAL_DEEPLAB = False


def print_time(start, task):
    end = timeit.default_timer()
    training_time = end - start
    minutes, seconds = divmod(int(training_time), 60)
    hours, minutes = divmod(minutes, 60)
    print('Time to complete {:s}:  {:d}:{:02d}:{:02d}'.format(task, hours, minutes, seconds))


if __name__ == '__main__':
    config = Config()

    if CREATE_SALIENCY:
        from saliency.create_saliency import main as create_saliency_main
        start = timeit.default_timer()
        print("Starting creating saliency images")
        create_saliency_main(config)
        print_time(start, "creating saliency images")

    if TRAIN_SEED:
        from seed.train_seed import main as train_seed_main
        start = timeit.default_timer()
        print("Starting training seeder")
        train_seed_main(config)
        print_time(start, "training seeder")

    if CREATE_ATTENTION:
        from seed.create_network_attention import main as create_attention_main
        start = timeit.default_timer()
        print("Starting attention generation")
        create_attention_main(config)
        print_time(start, "attention generation")

    if CREATE_GUIDE:
        from seed.create_guides import main as create_guide_main
        start = timeit.default_timer()
        print("Starting creating guides")
        create_guide_main(config)
        print_time(start, "creating guides")

    if TRAIN_DEEPLAB:
        from segment.train_deeplabv3 import main as segment_train
        start = timeit.default_timer()
        print("Starting training segmenter")
        segment_train(config)
        print_time(start, "training segmenter")

    if EVAL_DEEPLAB:
        from segment.evaluate_deeplabv3 import main as segment_eval
        start = timeit.default_timer()
        print("Starting testing segmenter")
        segment_eval(config)
        print_time(start, "testing segmenter")
