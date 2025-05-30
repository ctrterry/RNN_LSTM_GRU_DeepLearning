# RNN_LSTM_GRU_DeepLearning

## Datasets: 
- I can't upload the classification dataset. Due to extramly biggest.

## How to run my code?

### Run Text Classificatino

- Step1: download Classificatino datasets from IFM lab and put dataset into the data foler. Such as data/stage_4_data/text_classification
- Step2: cd Stage_4_done
- Step3: python script/stage_4_script/classification/main.py
[Classification_Result](Stage_4_done/result/stage_4_result/classification)
- Step4: python script/stage_4_script/classification/ablation_studies.py  (Notes: This will running over 4 hours, we are using GPU. Extraly long time).
[Alation Studies_Result.txt](output.txt)
---

### Run Text Generation
- Step1: cd Stage_4_done
- Step2: python script/stage_4_script/text_generation/main.py
[RNN_result](Stage_4_done/result/stage_4_result/text_generation/rnn)
- Step3: python script/stage_4_script/text_generation/train_lstm.py
- [LSTM_result](Stage_4_done/result/stage_4_result/text_generation/lstm)
- Step4: python script/stage_4_script/text_generation/train_gru.py
[GRU_result](Stage_4_done/result/stage_4_result/text_generation/gru)
---

- THis is my tree structure
- .
  - README.md
  - Stage_4_done
    - data
      - stage_4_data
        - text_classification
        - text_generation
          - check_dataset.ipynb
          - data
          - data.csv
          - ReadMe.docx
    - local_code
      - stage_4_code
        - __init__.py
        - __pycache__
          - __init__.cpython-311.pyc
        - classification
          - __init__.py
          - __pycache__
            - __init__.cpython-311.pyc
            - data_loader.cpython-311.pyc
            - evaluation_accuracy.cpython-311.pyc
            - evaluation_plot.cpython-311.pyc
            - method_RNN.cpython-311.pyc
          - data_loader.py
          - evaluation_accuracy.py
          - evaluation_plot.py
          - method_RNN.py
        - text_generation
          - __pycache__
            - data_loader.cpython-311.pyc
            - evaluation_plot.cpython-311.pyc
            - Method_GRU.cpython-311.pyc
            - Method_LSTM.cpython-311.pyc
            - Method_RNN.cpython-311.pyc
            - model.cpython-311.pyc
          - data_loader.py
          - evaluation_plot.py
          - Method_GRU.py
          - Method_LSTM.py
          - Method_RNN.py
          - model.py
    - result
      - stage_4_result
        - classification
          - metrics_plot.png
          - metrics.txt
          - training_test_loss.png
        - text_generation
          - gru
            - generated_text.txt
            - metrics_0520.png
            - metrics_0520.txt
            - perplexity_0520.png
            - training_progress_0520.png
          - lstm
            - generated_text.txt
            - metrics_0520.png
            - metrics_0520.txt
            - perplexity_0520.png
            - training_progress_0520.png
          - rnn
            - generated_text.txt
            - metrics_0520.png
            - metrics_0520.txt
            - perplexity_0520.png
            - training_progress_0520.png
    - script
      - stage_4_script
        - classification
          - ablation_studies.py
          - main.py
        - text_generation
          - main.py
          - plot_model_architecture.py
          - train_gru.py
          - train_lstm.py
