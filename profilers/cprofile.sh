rm /tmp/profile.pstats
python3 -m cProfile -o /tmp/profile.pstats $HOME/neuralhydrology/neuralhydrology/nh_run.py train --config-file $HOME/neuralhydrology/config/multimet_mean_embedding_forecast_lstm.yml
gprof2dot -f pstats /tmp/profile.pstats | dot -Tpng -o /tmp/profile.png
