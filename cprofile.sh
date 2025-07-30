python3 -m cProfile -o /tmp/profile.pstats /usr/local/google/home/amarkel/neuralhydrology/neuralhydrology/nh_run.py train --config-file config/multimet_mean_embedding_forecast_lstm.yml
gprof2dot -f pstats /tmp/output_filename.pstats | dot -Tpng -o /tmp/image_output.png
