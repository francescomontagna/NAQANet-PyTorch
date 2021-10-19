"""
Format custom annotated data as DROP

DROP sample:
{
    "{id, e.g. nfl1984}": {
        "passage: "...",
        "qa_pairs": [
            {
                "question": "Who scored the first touchdown of the game?",
                "answer": {
                    "number": "",
                    "date": {
                        "day": "",
                        "month": "",
                        "year": ""
                    },
                    "spans": [
                        "Chaz Schilens"
                    ]
                },
                "query_id": "f37e81fa-ef7b-4583-b671-762fc433faa9",
                "validated_answers": [
                    {
                        "number": "",
                        "date": {
                            "day": "",
                            "month": "",
                            "year": ""
                        },
                        "spans": [
                            "Chaz Schilens"
                        ]
                    },
                    {
                        "number": "",
                        "date": {
                            "day": "",
                            "month": "",
                            "year": ""
                        },
                        "spans": [
                            "JaMarcus Russell"
                        ]
                    }
                ]
            }
        ]
    
    }

}


QUERIES (absolutely review):
- "population": ["dead", "injured", "missing", "evacuated", "recovered", "hospitalized", "rescued"]
  
  recovered: only 3 records
  hospitalized: only 4 records
  evacuated: only 4 records



- "infrastructure": ["residential", "water_network", "road", "power_network", "bridge"]
  bridge: only 5 records (all with 'bridge' in the passage)
  water_network: 0 records
  road, power_network: 0 records != from 0

"""
import json
from dataset_utils import naqanet_format

def main():
    path = "annotated-data/validation_tweets_gt.json"
    records = []
    with open(path, 'r') as file:
        records = json.load(file, encoding="utf-8")

    drop_formatted = dict()
    for i, record in enumerate(records):
        drop_formatted[f"annotated_{i}"] = naqanet_format(record)
        break
    print(drop_formatted)
main()
