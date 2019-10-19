#!/usr/bin/env bash

python a3_link_prediction.py -t real -d football
python a3_link_prediction.py -t real -d polblogs
python a3_link_prediction.py -t real -d polbooks
python a3_link_prediction.py -t real -d strike
python a3_link_prediction.py -t real -d karate

python a3_link_prediction.py -t lfr -d lfr

python a3_link_prediction.py -t node -d citeseer
python a3_link_prediction.py -t node -d cora
python a3_link_prediction.py -t node -d pubmed

python a3_node_classification.py -t real -d football
python a3_node_classification.py -t real -d polblogs
python a3_node_classification.py -t real -d polbooks
python a3_node_classification.py -t real -d strike
python a3_node_classification.py -t real -d karate

python a3_node_classification.py -t lfr -d lfr

python a3_node_classification.py -t node -d citeseer
python a3_node_classification.py -t node -d cora
python a3_node_classification.py -t node -d pubmed