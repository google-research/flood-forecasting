rm -f /tmp/memray_output.bin /tmp/memray_flamegraph.html

echo Profiling...
memray run --aggregate --native -o /tmp/memray_output.bin ~/neuralhydrology/neuralhydrology/nh_run.py \
        train --config-file ~/neuralhydrology/config/multimet_mean_embedding_forecast_lstm.yml

echo Analysing...
memray flamegraph -o /tmp/memray_flamegraph.html /tmp/memray_output.bin

open /tmp/memray_flamegraph.html
