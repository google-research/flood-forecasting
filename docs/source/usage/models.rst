Modelzoo
========

This section provides an overview of the forecasting models available in the library.

Model Heads
-----------
The head of the model is used on top of the model class and relates the outputs of the model to the predicted variable.

Regression
^^^^^^^^^^
:py:class:`googlehydrology.modelzoo.head.Regression` provides a single layer *regression* head, that includes different activation options for the output.

CMAL
^^^^
:py:class:`googlehydrology.modelzoo.head.CMAL` implements a *Countable Mixture of Asymmetric Laplacians* head. That is, a mixture density network with asymmetric Laplace distributions as components.

Model Classes
-------------

BaseModel
^^^^^^^^^
Abstract base class from which all models derive. Do not use this class for model training.

Handoff-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^
:py:class:`googlehydrology.modelzoo.handoff_forecast_lstm.HandoffForecastLSTM` is a forecasting model
that uses a state-handoff to transition from a hindcast sequence (LSTM)
model to a forecast sequence (LSTM) model. The hindcast model is run from the past up to present
(the issue time of the forecast) and then passes the cell state and hidden state of the LSTM into
a (nonlinear) handoff network, which is then used to initialize the cell state and hidden state of a
new LSTM that rolls out over the forecast period.

Mean-Embedding-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`googlehydrology.modelzoo.mean_embedding_forecast_lstm.MeanEmbeddingForecastLSTM` is a forecasting
model that uses separate embedding networks for hindcast and forecast inputs. It aggregates these inputs
using masked means before passing them into respective LSTMs for the hindcast and forecast periods.

Implementing a new model
^^^^^^^^^^^^^^^^^^^^^^^^
The listing below shows the skeleton of a template model you can use to start implementing your own model.

**Crucial Steps:**

1.  **Inherit from BaseModel:** Your class must inherit from :py:class:`googlehydrology.modelzoo.basemodel.BaseModel`.
2.  **Define module_parts:** You must define a list called ``module_parts`` containing the names of the sub-modules (e.g., LSTMs, Linear layers) in your class. This is required for the fine-tuning logic to know which parts of the model to freeze or unfreeze.
3.  **Register the Model:** Once implemented, you must modify :py:func:`googlehydrology.modelzoo.__init__.get_model` to instantiate your class when its name is found in the config.

.. code-block:: python

    import torch

    from googlehydrology.modelzoo.basemodel import BaseModel


    class TemplateModel(BaseModel):

        # The `module_parts` variable is a list of all of the different model components.
        # You must construct and name these components. This is necessary in order to freeze
        # and unfreeze individual components for fine tuning.
        module_parts = [...]

        def __init__(self, cfg: dict):
            """Initialize the model

            Each model receives as only input the config dictionary. From this, the entire model can be implemented.

            Each Model inherits from the BaseModel, which implements some universal functionality. The basemodel also
            defines the output_size, which can be used here as a given attribute (self.output_size)

            Parameters
            ----------
            cfg : dict
                Configuration of the run, read from the config file with some additional keys (such as number of basins).
            """
            super(TemplateModel, self).__init__(cfg=cfg)

            ###########################
            # Create model parts here #
            ###########################

        def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Forward pass through the model
          
            Parameters
            ----------
            - 'x_d_hindcast': Hindcast dynamic inputs 
              Shape: [batch, seq_length, features]
            - 'x_d_forecast': Forecast dynamic inputs 
              Shape: [batch, lead_time, features]
            - 'x_s': Static inputs 
              Shape: [batch, features]
            
            Returns
            -------
            dict[str, torch.Tensor]
                The dictionary must contain the key 'y_hat' with predictions.
                Shape: [batch, seq_length, num_targets]
            """

            ###############################
            # Implement forward pass here #
            ###############################
            
            # Example forward pass (remove when you add your own modeling logic)
            hindcast = data['x_d_hindcast']
            forecast = data['x_d_forecast']
            statics = data['x_s']
            
            # ... implement your forecasting logic here ...
            
            # Example placeholder output (remove when you add your own modeling logic)
            batch_size, seq_len, _ = hindcast.shape
            output = torch.zeros(batch_size, seq_len, self.output_size)

            return {'y_hat': output}