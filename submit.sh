#!/usr/bin/env bash

source venv/bin/activate
pip3 uninstall causaldag
pip3 install causaldag
pip3 freeze > requirements.txt
zip -r ../dct-submit.zip . -x ./data/\* -x ./figures/\* -x ./old_figures/\* -x ./scratch/\* -x ./.git/\* ./venv/\* ./.idea/\* *__pycache__/\* -x submit.sh
pip3 uninstall causaldag
pip3 install causaldag -e ~/Documents/projects/causaldag
