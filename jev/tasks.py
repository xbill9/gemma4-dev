"""The labelled tasks the comparison runs on.

Each task is one public test split with human labels and one question in the
Jev schema shape that structured_server.py parses. `gold` maps the dataset's
integer label to the option name the question offers, so a prediction is
scored by name and never by position.

Chosen to span the range a classifier sees: one near-ceiling task (sst2), one
clean multi-class task (ag_news), one with genuinely ambiguous labels (emotion)
and one hard yes/no task (irony), where calibration matters most.
"""

TASKS = {
    "sst2": {
        "dataset": "stanfordnlp/sst2",
        "config": "default",
        "split": "validation",  # the test split carries no labels
        "text_field": "sentence",
        "label_field": "label",
        "gold": {0: "negative", 1: "positive"},
        "question": {
            "id": "sentiment",
            "type": "choice",
            "instructions": "What is the sentiment of this movie review sentence?",
            "options": [
                {"name": "negative", "description": "unfavourable"},
                {"name": "positive", "description": "favourable"},
            ],
        },
    },
    "ag_news": {
        "dataset": "fancyzhx/ag_news",
        "config": "default",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "gold": {0: "world", 1: "sports", 2: "business", 3: "scitech"},
        "question": {
            "id": "topic",
            "type": "choice",
            "instructions": "Which section of a news site does this article belong in?",
            "options": [
                {"name": "world", "description": "world news and politics"},
                {"name": "sports", "description": "sports"},
                {"name": "business", "description": "business and economy"},
                {"name": "scitech", "description": "science and technology"},
            ],
        },
    },
    "emotion": {
        "dataset": "dair-ai/emotion",
        "config": "split",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "gold": {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"},
        "question": {
            "id": "emotion",
            "type": "choice",
            "instructions": "Which emotion does the writer of this message express?",
            "options": [
                {"name": "sadness"},
                {"name": "joy"},
                {"name": "love"},
                {"name": "anger"},
                {"name": "fear"},
                {"name": "surprise"},
            ],
        },
    },
    "irony": {
        "dataset": "cardiffnlp/tweet_eval",
        "config": "irony",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        # noul questions answer yes/no; the parser names the choices yes, no
        "gold": {0: "no", 1: "yes"},
        "question": {
            "id": "ironic",
            "type": "noul",
            "instructions": "Is this tweet ironic?",
            "criteria": {"true": "the tweet is ironic", "false": "the tweet is not ironic"},
        },
    },
}
