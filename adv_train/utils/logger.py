import json
import uuid
import os
from argparse import Namespace
import torch
from adv_train.model import load_classifier
import pickle
from collections import defaultdict
from enum import Enum
import pandas as pd


class Database():
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def create_record(self):
        _id = str(uuid.uuid4())
        record = Record(self, _id)
        record.set_state(RecordState.RUNNING)
        return record

    def load_all_records(self):
        list_records = {}
        list_id = os.listdir(self.log_dir)
        for _id in list_id:
            record = Record(self, _id)
            list_records[_id] = record
        return list_records

    def load_record(self, id):
        record = Record(self, id)
        return record

    def reset_state(self, state):
        all_records = self.load_all_records()
        for _id, record in all_records.items():
            state = record.get_state()
            if state == state:
                record.set_state(RecordState.COMPLETED)

    def extract_to_df(self):
        all_records = self.load_all_records()
        results = defaultdict(list)
        for _id, record in all_records.items():
            state = record.get_state()
            if state == RecordState.EVAL_DONE:
                for key, value in record.load_eval().items():
                    results[key].append(value)
                for key, value in record.load_hparams().items():
                    results[key].append(value)
                    
        df = pd.DataFrame.from_dict(results)
        return df


class RecordState(Enum):
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    EVAL_DONE = "eval_done"
    EVAL_RUNNING = "eval_running"
    EVAL_WAITING = "eval_waiting"


class Record():
    def __init__(self, db, id):
        self.id = id
        self.path = os.path.join(db.log_dir, self.id)
        os.makedirs(self.path, exist_ok=True)
        self.load()

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

    def _save(self, results, filename):
        with open(filename, "w") as f:
            json.dump(results, f, indent=1)

    def save(self):
        filename = os.path.join(self.path, "results.json")
        self._save(self.results, filename)

    def save_eval(self, results):
        filename = os.path.join(self.path, "eval.json")
        self._save(results, filename)
        self.set_state(RecordState.EVAL_DONE)

    def load_eval(self):
        results = defaultdict(list)
        filename = os.path.join(self.path, "eval.json")
        with open(filename, "r") as f:
            results = json.load(f)
        return results

    def load(self):
        self.load_hparams()
        results = defaultdict(list)
        filename = os.path.join(self.path, "results.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results = json.load(f)
        self.results = results
        return results

    def load_model(self, device=None, eval=True):
        filename = os.path.join(self.path, "model.pth")
        
        if self.hparams is None:
            self.haparams = self.load_hparams()
        
        model = load_classifier(
            self.hparams["dataset"],
            self.hparams["type"],
            model_path=filename,
            device=device,
            eval=eval,
        )

        return model

