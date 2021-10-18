"""
Format custom annotated data as DROP

{
    "id": {
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
- "population": ["Dead", "injured", "missing", "evacuated", "recovered", "hospitalized", "rescued"]
  
  recovered: only 3 records
  hospitalized: only 4 records
  evacuated: only 4 records



- "infrastructure": ["residential", "water_network", "road", "power_network", "bridge"]
  power_network: only 8 records ()
  bridge: only 5 records (all with 'bridge' in the passage)
  water_network

"""
import json
from dataset_utils import parse_record, get_records_string

def main():
    path = "annotated/validation_tweets_gt.json"
    records = []
    with open(path, 'r') as file:
        file.readline() # read '['
        while "]" not in file.readline():
            records.append(parse_record(get_records_string(file)))
            break

    print(records[0])

main()