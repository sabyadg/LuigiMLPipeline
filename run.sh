#!/usr/bin/env bash
#python pipeline.py ScoreTask --local-scheduler --tweet-file airline_tweets.csv
luigid --background --port=8082 --logdir=logs
python pipeline.py  --id=4515
#Please run by executing chmod +x on run.sh and then just type ./run.sh in terminal
#To visualize luigi scheduler just go to browser and type :http://localhost:8082/