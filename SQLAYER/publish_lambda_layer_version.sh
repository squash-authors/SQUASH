#!/usr/bin/bash

set -x

source ./sqvenv/bin/activate
cd sqlayer
python setup.py bdist_wheel sdist
pip install .
cd ../python
rm -r sqlayer
rm -r sqlayer-0.0.1.dist-info
cp -r ../sqvenv/lib/python3.11/site-packages/sqlayer .
cp -r ../sqvenv/lib/python3.11/site-packages/sqlayer-0.0.1.dist-info .
cd ..
rm squash-layer.zip
zip -r squash-layer.zip ./python
aws lambda publish-layer-version --layer-name squash-layer --zip-file fileb://squash-layer.zip --compatible-runtimes python3.11
set +x
