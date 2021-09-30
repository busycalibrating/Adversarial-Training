import json
import uuid
import os
from argparse import Namespace
import torch
from adv_train.model import load_classifier
import pickle
from collections import defaultdict
from enum import Enum


class Database():
    def __init__(self, log_dir="./logs"):
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.load_all_records()

    def create_record(self):
        _id = self._generate_uuid()
        record = Record(self, _id)
        self.list_records[_id] = record
        return record

    def _generate_uuid(self):
        while True:
            _id = str(uuid.uuid4())
            if _id not in self.list_records:
                break
        return _id 

    def load_all_records(self):
        self.list_records = {}
        list_id = os.listdir(self.log_dir)
        for _id in list_id:
            record = Record(self, _id)
            self.list_records[_id] = record

    def load_record(self, id):
        record = Record(self, id)
        return record


class RecordState(Enum):
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    EVAL_DONE = "eval_done"


class Record():
    def __init__(self, db, id):
        self.id = id
        self.path = os.path.join(db.log_dir, self.id)
        os.makedirs(self.path, exist_ok=True)
        self.load()
        self.set_state(RecordState.RUNNING)

    def set_state(self, state):
        filename = os.path.join(self.path, ".STATE")
        with open(filename, "w") as f:
            f.write(state.name)
            f.flush()

    def get_state(self):
        filename = os.path.join(self.path, ".STATE")
        with open(filename, "r") as f:
            state = RecordState[f.read()]
        return state

    def close(self):
        self.save()
        self.set_state(RecordState.COMPLETED)

    def fail(self):
        self.save()
        self.set_state(RecordState.FAILED)

    def save_hparams(self, hparams):
        if isinstance(hparams, Namespace):
            hparams = vars(hparams) 
        filename = os.path.join(self.path, "hparams.pkl")
        with open(filename, "wb") as f:
            pickle.dump(hparams, f)
            f.flush()
        self.hparams = hparams

    def load_hparams(self):
        hparams = None
        filename = os.path.join(self.path, "hparams.pkl")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                hparams = pickle.load(f)
        self.hparams = hparams
        return hparams

    def save_model(self, model):
        filename = os.path.join(self.path, "model.pth")
        torch.save(model.state_dict(), filename)

    def add(self, results):
        for key, value in results.items():
            self.results[key].append(value)

    def save(self):
        filename = os.path.join(self.path, "results.json")
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=1)

    def load(self):
        self.load_hparams()
        results = defaultdict(list)
        filename = os.path.join(self.path, "results.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results = json.load(f)
        self.results = results

    def load_model(self, device=None, eval=True):
        filename = os.path.join(self.path, "hparams.json")
        
        if self.hparams is None:
            self.haparams = self.load_hparams()
        
        self.model = load_classifier(
            self.hparams.dataset,
            self.hparams.type,
            model_path=filename,
            device=device,
            eval=eval,
        )

