"""
EcoPro Tree Mortality
Utility Functions
"""
import yaml


def load_config(configfile):
    with open(configfile, 'r') as f:
        return yaml.safe_load(f)
