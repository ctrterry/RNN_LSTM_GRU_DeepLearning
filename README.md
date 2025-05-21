# RNN_LSTM_GRU_DeepLearning

## Datasets: 
- since, I can't upload the classification dataset. Becuase this dataset is too big. But you can put this dataset into my data/stage_4_data area.
- But I have test_generation dataset on the github

## How to run my code?

### Run Text Classificatino

- Step1: download Classificatino datasets from IFM lab and put dataset into the data foler. Such as data/stage_4_data/text_classification
- Step2: cd Stage_4_done
- Step3: python script/stage_4_script/classification/main.py
- Step4: python script/stage_4_script/classification/ablation_studies.py  (Notes: This will running over 4 hours, we are using GPU. Extraly long time).

---

### Run Text Generation
- Step1: cd Stage_4_done
- Step2: python script/stage_4_script/text_generation/main.py
- Step3: python script/stage_4_script/text_generation/train_lstm.py
- Step4: python script/stage_4_script/text_generation/train_gru.py

---

- THis is my tree structure
- .
├── README.md
└── Stage_4_done
    ├── data
    │   └── stage_4_data
    │       ├── text_classification
    │       └── text_generation
    │           ├── check_dataset.ipynb
    │           ├── data
    │           ├── data.csv
    │           └── ReadMe.docx
    ├── local_code
    │   └── stage_4_code
    │       ├── __init__.py
    │       ├── __pycache__
    │       │   └── __init__.cpython-311.pyc
    │       ├── classification
    │       │   ├── __init__.py
    │       │   ├── __pycache__
    │       │   │   ├── __init__.cpython-311.pyc
    │       │   │   ├── data_loader.cpython-311.pyc
    │       │   │   ├── evaluation_accuracy.cpython-311.pyc
    │       │   │   ├── evaluation_plot.cpython-311.pyc
    │       │   │   └── method_RNN.cpython-311.pyc
    │       │   ├── data_loader.py
    │       │   ├── evaluation_accuracy.py
    │       │   ├── evaluation_plot.py
    │       │   └── method_RNN.py
    │       └── text_generation
    │           ├── __pycache__
    │           │   ├── data_loader.cpython-311.pyc
    │           │   ├── evaluation_plot.cpython-311.pyc
    │           │   ├── Method_GRU.cpython-311.pyc
    │           │   ├── Method_LSTM.cpython-311.pyc
    │           │   ├── Method_RNN.cpython-311.pyc
    │           │   └── model.cpython-311.pyc
    │           ├── data_loader.py
    │           ├── evaluation_plot.py
    │           ├── Method_GRU.py
    │           ├── Method_LSTM.py
    │           ├── Method_RNN.py
    │           └── model.py
    ├── result
    │   └── stage_4_result
    │       ├── classification
    │       │   ├── metrics_plot.png
    │       │   ├── metrics.txt
    │       │   └── training_test_loss.png
    │       └── text_generation
    │           ├── gru
    │           │   ├── generated_text.txt
    │           │   ├── metrics_0520.png
    │           │   ├── metrics_0520.txt
    │           │   ├── perplexity_0520.png
    │           │   └── training_progress_0520.png
    │           ├── lstm
    │           │   ├── generated_text.txt
    │           │   ├── metrics_0520.png
    │           │   ├── metrics_0520.txt
    │           │   ├── perplexity_0520.png
    │           │   └── training_progress_0520.png
    │           └── rnn
    │               ├── generated_text.txt
    │               ├── metrics_0520.png
    │               ├── metrics_0520.txt
    │               ├── perplexity_0520.png
    │               └── training_progress_0520.png
    └── script
        └── stage_4_script
            ├── classification
            │   ├── ablation_studies.py
            │   └── main.py
            └── text_generation
                ├── main.py
                ├── plot_model_architecture.py
                ├── train_gru.py
                └── train_lstm.py

