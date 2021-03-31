import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
ANALYSIS_DIR = os.path.join(ROOT_DIR, 'analysis')
if not os.path.isdir(ANALYSIS_DIR):
    os.mkdir(ANALYSIS_DIR)
sys.path.append(ROOT_DIR)
