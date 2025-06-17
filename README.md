# UATSAD
![License](https://img.shields.io/github/license/xl814/UATSAD)
![Stars](https://img.shields.io/github/stars/xl814/UATSAD?style=social)

This is the official implementation of the paper "Uncertainty-Aware Time Series Anomaly Detection
Based on Deep Ensembles" 
## Prerequisites
Make sure you have installed all of following prerequisites on yor machine:
- Node.js (we used version 22.14.0)
- Python (we used version 3.10)
## Installation
We suggest using a virtual environment, such as [virtualenv](https://virtualenv.pypa.io/en/latest/):
```bash
cd UATSAD
virtualenv .vene
source .vene/bin/activate
pip install -r requirements.txt
```

## Result Reproduction
1. Toy experiment (For the related implementation, please referto `src/`):
    - reproduce the **experiment 1** in the main text:
        ```bash
        ./scripts/toy_experiment.sh
        ```
        we provide a notebbok to visualize the toy example result in `src/toy_example.ipynb`.
    - the result of hyperparameter sensitivity analysis experiments:
        ```bash
        ./scripts/toy_experiment.sh sumplement
        ```
2. Uncertainty calibration experimens reported in the paper 
    ```bash
    ./scripts/run.sh
    ```

    We also provide a notebook to visualize the uncertainty calibration curves in `src/epic_alea_evaluation.ipynb`.

## Run the Application
### 1. Run the server
- Take `SMAP-P1` as an example, we first train the model :
    ```bash
    python ./server/process_model.py 
    ```
- And, load the trained model to run the server:
    ```bash
    fastapi dev server/main.py
    ```
### 2. Launch UATSAD
```bash
    cd react-app
    npm run dev
```
## Cite this work

## License