def generate_id(record):
    """
    Generate a unique ID (if needed by the NN)
    """
    return None


def format_answer(answer_dict):
    """
    Add date and spans to the answer record
    """
    answer_dict['date'] = dict()
    for x in ('day', 'month', 'year'):
        answer_dict['date'][x] = ""

    answer_dict['spans'] = []

    return answer_dict


def naqanet_format(record):
    """
    record (dict): example of input record
    {
      "created_at": "2021-05-22T00:39:30.000Z",
      "event_id": 20498,
      "informative": true,
      "lang": "en",
      "text": "Series Of Strong Earthquakes Rattle China; 2 Dead, 22 Injured, Houses\u00a0Damaged https://t.co/x7Ps4Abg9x",
      "tweet_id": 1395932302333693957,
      "impact": {
         "population": {
            "dead": 2,
            "injured": 22
         },
         "infrastructures": {
            "residential": 0
         }
      }
    }
    """
    naqanet_record = dict()
    queries = { # TODO: review queries: the question highly influence the accuracy
        'population': {
            'dead': 'How many people dead?',
            'injured': 'How many people injured?',
            'missing': 'How many people missing?',
            'evacuated': 'How many people rescued?',
            'recovered': 'How many people recovered?',
            'hospitalized': 'How many people evacuated?',
            'rescued': 'How many people hospitalized?',
        },

        # Per infrastructure la vedo moooolto dura che riesca a rispondere.
        'infrastructures': {
            'residential': 'How many buildings damaged?',
            'bridge': 'How may bridges damaged?'
        }
    } 

    naqanet_record['passage'] = record['text']
    naqanet_record['qa_pairs'] = list()

    for type_damage in queries.keys(): # population, infrastructure
        for tag, query in queries[type_damage].items(): # e.g. 'dead', 'How many people dead?'
            qa_pair = dict()
            qa_pair['question'] = query
            qa_pair['answer'] = dict()
            try:
                qa_pair['answer']['number'] = record['impact'][type_damage][tag]
            except KeyError: 
                qa_pair['answer']['number'] = ""

            format_answer(qa_pair['answer'])    

            qa_pair['query_id'] = generate_id(record)
            qa_pair['validated_answer'] = qa_pair['answer'] # Check

            naqanet_record['qa_pairs'].append(qa_pair)

    return naqanet_record