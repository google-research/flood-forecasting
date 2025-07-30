python3 -m cProfile -o /tmp/profile.pstats /usr/local/google/home/amarkel/neuralhydrology/neuralhydrology/nh_run.py train --config-file config/multimet_mean_embedding_forecast_lstm.yml
gprof2dot -f pstats /tmp/profile.pstats | dot -Tpng -o /tmp/profile.png
