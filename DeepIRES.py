import numpy as np
import pandas as pd
import argparse
from model.model import deepires_model
from model.sequence_encode import get_data_onehot
import warnings
from Bio import SeqIO
import os


def read_fa(path):
    res = {}
    records = list(SeqIO.parse(path, format='fasta'))
    for x in records:
        id = str(x.id)
        seq = str(x.seq)
        res[id] = seq
    return res



def predict_score(data):
    test = []
    start = []
    stop = []
    model = deepires_model()
    model.load_weights('weights/first').expect_partial()
    for seq in data:
        if (len(seq)) > 174:
            score = []
            i = 1
            while i + 173 <= len(seq):
                seqq = np.array(seq[i - 1:i + 173]).reshape(1, )
                x = get_data_onehot(seqq, maxlen=174)
                score.append(model.predict(x,verbose=0)[0][0])
                i = i + 50
            seqlast = np.array(seq[-174:]).reshape(1, )
            x1 = get_data_onehot(seqlast, maxlen=174)
            score.append(model.predict(x1,verbose=0)[0][0])
            max_score = max(score)
            max_index = score.index(max_score)
            test.append(max_score)
            if max_score == score[-1]:
                start.append(len(seq) - 173)
                stop.append(len(seq))
            else:
                startt = 50 * max_index + 1
                start.append(startt)
                stop.append(startt + 173)
        else:
            seqq = np.array(seq).reshape(1, )
            x = get_data_onehot(seqq, maxlen=174)
            test.append(model.predict(x,verbose=0)[0][0])
            start.append(1)
            stop.append(len(seq))
    return test, start, stop
def DeepIRES_predict():
    print('prediction start \n')
    print('------------------------------------------------------------------')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_file', default=None, metavar='', type=str, required=True, help='please make sure your input is .fasta fomat ')
    #parser.add_argument('-b','--batch_size', default=32, help='Batch size (default=32),This parameter affects the speed of the prediction', metavar='', type=int)
    parser.add_argument('-o', '--out', default=None,  metavar='',type=str,required=True, help='assign your output file')
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    name_t = []
    seq_t = []
    res = read_fa('./data/' + f'{args.input_file}')
    for name in res.keys():
        name_t.append(name)
        seq_t.append(res[name].replace('U', 'T'))
    score, start, stop = predict_score(seq_t)
    out_dir = './result'
    out_name = args.out
    outfile = f'{out_dir}/{out_name}.csv'
    datas = {'name':name_t,"score": score, "start": start, "stop": stop}
    data = pd.DataFrame(datas)
    data.to_csv(outfile, index=False)
    print('------------------------------------------------------------------')
    print('prediction complete \n')
    print('The results were saved in {}'.format(outfile))
if __name__ == "__main__":
    DeepIRES_predict()
