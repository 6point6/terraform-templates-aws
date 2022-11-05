import os
import pandas as pd
import io



def load_data(ROOT, verbose=True):
    with open(os.path.join(ROOT, 'USCensus1990raw.attributes.txt'), 'r') as f:
        text = f.read()
    if verbose:
        print(text)

    sep = '__________________________________________________________________________________'

    all_codes = {}
    for t in text.split(sep)[1:]:

        lines = t.split('\n')

        colname = lines[1][:10].rstrip(' ')
        lines = [l.lstrip(' ').rstrip(' ') for l in lines[2:] if (l != '') and not l.startswith('VAR:')]
        coding = {}
        for line in lines:
            l = [l for l in line.split(' ') if l != '']
            coding[l[0]] = " ".join(l[1:])

        all_codes[colname] = coding

    cols = text.split('__________________________________________________________________________________')

    cols = [c.split('\n')[1] for c in cols[1:]]
    cols = [c for c in cols if c != '']
    frame = ""
    for c in cols:
        li = [c[:10].rstrip(' '), c[13:14], c[21:22], c[28:29], c[42:].replace(",", ".")]
        if len(li) != 5:
            if verbose:
                print(li)
        frame += r",".join(li) + "\n"

    df_meta = pd.read_csv(io.StringIO(frame), sep=',', header=None, names=['Var', 'Type', 'Des', 'Len', 'Description'])

    return df_meta, all_codes
