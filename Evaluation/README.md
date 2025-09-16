### Task 1: `results_spotting.json`

For the action spotting task, each json file needs to be constructed as follows:

```json
{
    "UrlLocal": "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal",
    "predictions": [ # list of predictions
        {
            "gameTime": "1 - 0:31", # format: "{half} - {minutes}:{seconds}",
            "label": "Ball out of play", # label for the spotting,
            "position": "31500", # time in milliseconds,
            "half": "1", # half of the game
            "confidence": "0.006630070507526398", # confidence score for the spotting,
        },
        {
            "gameTime": "1 - 0:39",
            "label": "Foul",
            "position": "39500",
            "half": "1",
            "confidence": "0.07358131557703018"
        },
        {
            "gameTime": "1 - 0:55",
            "label": "Foul",
            "position": "55500",
            "half": "1",
            "confidence": "0.20939764380455017"
        },
        ...
    ]
}
```
## How to evaluate locally the performances on the testing set

### Task 1: Spotting

```python
from SoccerNet.Evaluation.ActionSpotting import evaluate
results = evaluate(SoccerNet_path=PATH_DATASET, Predictions_path=PATH_PREDICTIONS,
                   split="test", version=2, prediction_file="results_spotting.json")

print("Average mAP: ", results["a_mAP"])
print("Average mAP per class: ", results["a_mAP_per_class"])
print("Average mAP visible: ", results["a_mAP_visible"])
print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("Average mAP unshown: ", results["a_mAP_unshown"])
print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])
```
