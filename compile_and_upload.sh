#!/bin/bash
pytest -v ./ # first test
#pdoc --html --force --output-dir ./docs micropolarray
cd docs/
./create_documentation.sh
cd ..
echo "Ended documentation, starting compiling..."
#pip-compile pyproject.toml --resolver=backtracking
#pipreqs --savepath=requirements.in
pip-compile pyproject.toml --resolver=backtracking --output-file requirements.txt -v
#echo "Ended compiling, starting build..."
#python3.12 -m build
#echo "Ended build, starting upload..."
#twine upload --verbose --skip-existing dist/*