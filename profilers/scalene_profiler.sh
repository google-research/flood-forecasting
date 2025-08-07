rm -f /tmp/scalene_output.html
echo Be patient after done running.
scalene --profile-all --stacks --web --outfile /tmp/scalene_output.html $HOME/neuralhydrology/neuralhydrology/nh_run.py --- train --config-file $HOME/neuralhydrology/config/multimet_mean_embedding_forecast_lstm.yml
open /tmp/scalene_output.html
