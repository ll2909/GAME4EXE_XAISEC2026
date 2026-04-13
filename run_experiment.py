from configparser import ConfigParser
from generate_adversarials import generate_adversarials
from transferability_test import transferability_test
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    print("Config path: ", path)
    cfg_parser = ConfigParser()
    cfg_parser.read(path)

    config = dict(cfg_parser["CONFIG"])

    generate_adversarials(config)
    transferability_test(config)
