from adv_train.utils.logger import Database, RecordState
from .test_langevin import LangevinAttack, Attacker
from collections import defaultdict
    

if __name__ == "__main__":
    parser = Attacker.add_arguments()
    parser = LangevinAttack.add_arguments(parser)
    parser.add_argument("--log_dir", default="./logs", type=str)

    args = parser.parse_args()

    args.eps_iter = 0.01
    attacker = [Attacker.FGSM, Attacker.PGD_40, Attacker.PGD_100]

    db = Database(args.log_dir)
    eval_process = LangevinAttack(args)
    eval_process.set_attacker(attacker)
    state_dict = defaultdict(int)
    for _id, record in db.load_all_records().items():
        state = record.get_state()
        state_dict[state] += 1
        if state == RecordState.COMPLETED:
            eval_process.load(record)
            record.set_state(RecordState.EVAL_WAITING)
            eval_process.run()
    
    print(state_dict)


