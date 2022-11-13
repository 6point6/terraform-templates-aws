import os
import pandas as pd
import io

STATES = ['Alaska',
          'California',
          'Hawaii',
          'Idaho',
          'Nevada',
          'Oregon',
          'Washington',
          'Arizona',
          'Arkansas',
          'Colorado',
          'Iowa',
          'Kansas',
          'Louisiana',
          'Minnesota',
          'Missouri',
          'Montana',
          'Nebraska',
          'New Mexico',
          'North Dakota',
          'Oklahoma',
          'South Dakota',
          'Texas',
          'Utah',
          'Wyoming',
          'Alabama',
          'Connecticut',
          'Delaware',
          'District of Columbia',
          'Florida',
          'Georgia',
          'Illinois',
          'Indiana',
          'Kentucky',
          'Maine',
          'Maryland',
          'Massachusetts',
          'Michigan',
          'Mississippi',
          'New Hampshire',
          'New Jersey',
          'New York',
          'North Carolina',
          'Ohio',
          'Pennsylvania',
          'Rhode Island',
          'South Carolina',
          'Tennessee',
          'Vermont',
          'Virginia',
          'West Virginia',
          'Wisconsin']

MAP_NON_COUNTRY_TO_CONTINENT = {
    'Korea, Not Specified': 'AS',
    'Germany, Not Specified': 'EU',
    'Union of Soviet Soc.ist Repub.s U.S.': 'EU',
    'Yugoslavia': 'EU',
    'West Germany': 'EU',
    'East Germany': 'EU',
    'West Berlin': 'EU',
    'Palestine, Not Specified': 'AS',
    'West Bank': 'AS',
    'Middle East, Not Specified': 'AS',
    'Indochina, Not Specified': 'AS',
    'Asia Minor, Not Specified': 'AS',
    'Europe, Not Specified': 'EU',
    'Central America, Not Specified': 'SA',
    'Scotland': 'EU',
    'United Kingdom, Not Specified': 'EU',
    'South America, Not Specified': 'SA',
    'Northern Ireland': 'EU',
    'Eastern Africa, Not Specified': 'AF',
    'Central Africa, Not Specified': 'AF',
    'Africa, Not Specified': 'AF',
    'North America, Not Specified': 'NA',
    'St. Vincent and the Grenadines': 'NA',
    'Dominican Repub.': 'NA',
    'Czechoslovakia': 'EU',
    'England': 'EU',
    'Burma': 'AS',
    'Azores Islands': 'EU',
    'Madeira Islands': 'EU',
    'Caribbean, Not Specified': 'NA',
    'Asia, Not Specified': 'AS',
    'Western Africa, Not Specified': 'AF',
    'Yemen, Peoples Democratic Repub.': 'AS',
    'Yemen Arab Repub.': 'AS',
    'Oceania, Not Specified': 'OC',
    'British West Indies, Not Specified': 'NA',
    'West Indies, Not Specified': 'NA',
    'Pitcairn Islands': 'OC',
    'Western Samoa': 'OC',
    'U.S. Virgin Islands': 'NA',
    'Netherlands Antilles': 'NA',
    'Wales': 'EU',
    'St. Kitts Nevis': 'NA',
    'Polynesia, Not Specified': 'OC'
}

MAP_NON_COUNTRY_TO_COUNTRY = {
    'Korea, Not Specified': 'South Korea',

    'Germany, Not Specified': 'Germany',
    'West Germany': 'Germany',
    'East Germany': 'Germany',
    'West Berlin': 'Germany',

    'Union of Soviet Soc.ist Repub.s U.S.': 'Russia',

    'Palestine, Not Specified': 'Palestine',
    'West Bank': 'Palestine',

    'St. Vincent and the Grenadines': 'St. Vincent and The Grenadines',
    'St. Kitts Nevis': 'Saint Kitts and Nevis',
    'Dominican Repub.': 'Dominican Republic',
    'Czechoslovakia': 'Czech Republic',

    'Burma': 'Myanmar',

    'Azores Islands': 'Portugal',
    'Madeira Islands': 'Portugal',

    'Yemen, Peoples Democratic Repub.': 'Yemen',
    'Yemen Arab Repub.': 'Yemen',

    'Pitcairn Islands': 'Pitcairn',
    'Western Samoa': 'Samoa',
    'U.S. Virgin Islands': 'Virgin Islands, U.S.',

    'Northern Ireland': 'UK',
    'England': 'UK',
    'Wales': 'UK',
    'Scotland': 'UK',
    'United Kingdom, Not Specified': 'UK',

    'Netherlands Antilles': 'Unknown',
    'British West Indies, Not Specified': 'Unknown',
    'West Indies, Not Specified': 'Unknown',
    'Western Africa, Not Specified': 'Unknown',
    'South America, Not Specified': 'Unknown',
    'North America, Not Specified': 'Unknown',
    'Eastern Africa, Not Specified': 'Unknown',
    'Central Africa, Not Specified': 'Unknown',
    'Africa, Not Specified': 'Unknown',
    'Asia, Not Specified': 'Unknown',
    'Caribbean, Not Specified': 'Unknown',
    'Oceania, Not Specified': 'Unknown',
    'Yugoslavia': 'Unknown',
    'Middle East, Not Specified': 'Unknown',
    'Indochina, Not Specified': 'Unknown',
    'Asia Minor, Not Specified': 'Unknown',
    'Europe, Not Specified': 'Unknown',
    'Central America, Not Specified': 'Unknown',
    'Polynesia, Not Specified': 'Unknown'
}


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
