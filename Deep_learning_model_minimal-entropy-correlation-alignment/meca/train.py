import os
import subprocess
import time
import argparse

py_filepath = 'main.py'
def main(method, alpha, ids):
    for (sid,tid) in ids:
        if sid == tid: continue
        bt = time.time()
        argline = ["python3", py_filepath, "--mode", "train",
                   "--method", method, "--alpha", str(alpha), "--sid", str(sid), "--tid", str(tid)]
        while True:
            proc = subprocess.run(argline)
            if (proc.returncode == 0):
                break
        print('Return Code:', proc.returncode)
        print('Time:',time.time()-bt)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', '-m', type=str, default='log-d-coral',
                    help='The method used to train the model.')
    ap.add_argument('--alpha', '-a', type=float, default=100,
                    help='The alpha value.')
    ap.add_argument('--sid', '-s', type=int, default=1,
                    help='The source ID.')
    ap.add_argument('--tid', '-t', type=int, default=2,
                    help='The target ID.')
    args = ap.parse_args()

    main(args.method, args.alpha, [(args.sid, args.tid)])