from configparser import ConfigParser
from generate_adversarials import generate_adversarials
from transferability_test import transferability_test

if __name__ == "__main__":
    
    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")

    config = dict(cfg_parser["CONFIG"])

    generate_adversarials(config)
    transferability_test(config)
