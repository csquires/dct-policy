#!/usr/bin/env bash

source venv/bin/activate
pip3 freeze > requirements.txt
zip -r submit.zip . -x data -x figures -x old_figures -x scratch -x .git/
