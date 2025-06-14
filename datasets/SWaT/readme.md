## Swat dataset from "A Dataset to Support Research in the Design of Secure Water Treatment Systems"
- We only use `LIT101` sensors to serve as train or test dataset
- We split train, valid, test:  4000-16000 16000-18000 and 0-4000
- label [(351, 538), (614, 702), (984, 1060), (1292, 1369), (1451, 1490), (1541, 1626), (2282, 2474), (3076, 3220)]
```python
    import pandas as pd
    import utils
    test_df = pd.read_csv('datasets/SWaT/SWaT_test.csv')
    test_np = test_df['LIT101'].to_numpy()
    test_label = test_df['label'].to_numpy()
    anomaly_index = utils.get_anomaly_segment(test_label)
    df_test = pd.DataFrame(test_np[0: 4000], columns=['value'])
    df_test.to_csv('swat_test.csv', index=False)

    df_train = pd.DataFrame(test_np[4000: 16000], columns=['value'])
    df_train.to_csv('swat_train.csv', index=False)
    print(anomaly_index)
```
