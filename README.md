# Medical appointements case

Code to predict whether or not a person that made a medical appointment will show up or not

## Src folder

| Name           | Description                                  |
| -------------- | -------------------------------------------- |
| **dataloader** | All data preprocessing functions live here.  |
| **trainer**    | Training / Evaluation of classifiers.        |
| **utils**      | Others functions utilities (Plotting etc...) |

# Data

Authentificate to kaggle API by downloading kaggle.json under {HOME} directory.
Once it's done, run this script to download the data under /data folder:
`
source download_data.sh
`

# Additional info

Precommit is enabled (.pre-commit-config.yaml) to run flake8 linting, mypy and pydocstyle tests before committing.
