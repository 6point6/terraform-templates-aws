import os
import pandas as pd
import io
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn

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


# ====================================================================================================================
# Helpers for High Level Demo
# ====================================================================================================================


    
def combine_rare_classes(df, col, r=0.1, map_to='other'):
    
    gb = df[[col]].copy()
    gb['count'] = 1
    gb = gb[[col, 'count']].copy().groupby(col).sum().reset_index().sort_values(col, ascending=False)
    gb['count'] = 100*gb['count'] / sum(gb['count'])
    
    mapper = {k:map_to for k in gb[gb['count'] < r][col].values}

    return mapper, gb


def load_data_hl(fpath              = 'us_census_data_cleaned_5.csv',
              target             = 'REARNING', 
              age_range          = [18,90],
              wage_range         = [5000,140000],
              drop_uninformative = True,
              retain: int        = 100):
    
    df = pd.read_csv(fpath).drop('Unnamed: 0', axis=1)
    # provide some sensible bounds

    TARGET = target

    # exclude irrelevant data
    assert age_range[0] < age_range[1]
    df = df[df['AGE']      > age_range[0]]
    df = df[df['AGE']      < age_range[1]]

    assert wage_range[0] < wage_range[1]
    df = df[df['REARNING'] > wage_range[0]]
    df = df[df['REARNING'] < wage_range[1]]
    
    mapper, gb = combine_rare_classes(df, 'RACE', 2.0, map_to = 'Other Race 700 799, 986 999')
    df['RACE'] = df['RACE'].apply(lambda x: mapper.get(x, x))
    
    if drop_uninformative:
        uninformative_columns = []
        for c in df.columns:
            if len(df[c].unique()) == 1:
                print(c)
                uninformative_columns += [c]
        df = df.drop(uninformative_columns, axis=1)

    if retain<100:
        df = df.iloc[:int(retain * df.shape[0]),:]
    
    return df

def _prepare_data(df, 
                  target  = 'REARNING', 
                  cols        = None, 
                  cat_cols    = None, 
                  encode_cats = True,
                  verbose     = True):
    
    if cols is None:
        cols = df.columns
    
    categorical_features = []
    if cat_cols is not None:
        categorical_features = cat_cols #[c for c in cats if c in cols]

    valid_cols = [c for c in cols if c in df.columns]
    df_X = df[valid_cols].copy()
    for c in df_X.columns:
        df_X[c] = df_X[c].astype(float, errors='ignore')
    assert target in df_X.columns

    feature_names = [x for x in list(df_X.columns) if x!=target]

    # ordinal encode the categorical features
    le_fitted ={}
    if encode_cats:
        for feature in categorical_features:
            le                 = preprocessing.LabelEncoder()
            le_fitted[feature] = le.fit(df_X[feature].values)
            df_X[feature]      = le_fitted[feature].transform(df_X[feature])
    
        if verbose and len(categorical_features)>0:
            print("Cardinalities")
            print("=============")
            for c in categorical_features:
                print(f"{c:<25} : {len(le_fitted[c].classes_)}")
            
    return df_X, le_fitted

def _split_data(df_X, target, scaler=None, shuffle=True):
    
    X = np.array(df_X.drop(target, axis=1), dtype=float)
    y = np.array(df_X[target], dtype=float)
    
    if shuffle:
            X, y = sklearn.utils.shuffle(X, y)

    n = X.shape[0]
    n_test = int(n*0.05)
    
    if scaler:
        scaler.fit(X)
        X = scaler.transform(X)

    X_train = X[n_test:,:]
    y_train = y[n_test:]

    X_test = X[:n_test,:]
    y_test = y[:n_test]
    
    return X_train, y_train, X_test, y_test


def _regression_metrics(model, X_test, y_test, verbose=True):
    results = {}
    # how did we do?
    results['score'] =model.score(X_test, y_test) 
    if verbose:
        print(f"Score                          : {results['score']:.2f}")

    # the mean error on prediction
    results['mean_abs_err'] = np.abs(model.predict(X_test) - y_test).mean()
    if verbose:
        print(f"Average Error                  : ${results['mean_abs_err']:.0f}")

    r = (model.predict(X_test) > 50000) == (y_test > 50000)
    results['class_err'] = 100 * sum(r) / len(r)
    if verbose:
        print(f"Classification (>$50,000) acc. : {results['class_err']:.2f}")
    
    return results


from typing import Optional

def _validate_features(df, 
                       target_feature: str, 
                       keep_features: Optional[list] = None, 
                       drop_features: Optional[list] = None):
    
    all_features = df.columns
   
    if keep_features is None:
        # default to keep all features
        keep_features = all_features
    if drop_features is None:
        # default to drop nothing
        drop_features = []

    if target_feature not in all_features:
        print(f"WARNING : target feature {target_feature} not found in data")
        raise RuntimeError(f"WARNING : target feature {target_feature} not found in data")
        
    keep_f = list(set(keep_features))
    drop_f = list(set(drop_features))
        
    if target_feature in keep_f:
        print(f"WARNING : target feature {target_feature} is in keep_feature list, it is not an input feature")
        keep_f.remove(target_feature)
        
    if target_feature in drop_f:
        print(f"WARNING : target feature {target_feature} is in drop_features list, it cannot be dropped")
        drop_f.remove(target_feature)
    
    for feature in drop_f:
        if feature in keep_f:
            print(f"WARNING : keep feature {feature} is also in drop_features!")
            keep_f.remove(feature)
    
    for feature in keep_f:
        if feature not in all_features:
            print(f"WARNING : keep feature {feature} is not in the data!")
            keep_f.remove(feature)
            
    df = df[keep_f + [target_feature]]
    
#     if len(drop_f)>0:
#         df = df.drop(drop_f, axis=1)
    
    features = list(df.columns)
#     features.remove(target_feature)
    
    return df, features
    

            
def reduce_cardinality(df, feature='OCCUP', group_below=700, new_value='other'):
    
    gb = df.groupby(feature).count().sort_values('AGE')['AGE']
    
    print(f"Old cardinality {gb.shape[0]}")

    low_occup_codes = set(list(gb[gb < group_below].index))
    repl = new_value # min(low_occup_codes)

    df_2 = df.copy()
    df_2[feature] = df[feature].apply(lambda x: repl if x in low_occup_codes else x)
    
    percentage_removed = 100 * sum(df_2[feature] == new_value)/df_2.shape[0]
    
    print(f"New cardinality {len(df_2[feature].unique())}")
    print(fr"{len(low_occup_codes)}/{len(df[feature].unique())} classes ({percentage_removed:.2f}% data) reduced to class {new_value}")
    
    return df_2


def train_linear_regression(df, 
                            target_feature,
                            keep_features = [],
                            drop_features = [],
                            dummy_features = [],
                            model_name="linearRegression",
                            model_params={},
                            reduce_cardinality=False,
                            n_repeats=10, 
                            test_split=0.05,
                            encode_cats=True,
                            scale=True,
                            verbose=True,NUM_FEATURES=[], df_meta=None):
    
    df_input, features = _validate_features(df, target_feature, keep_features, drop_features)
    if verbose:
        print_features(df, NUM_FEATURES, df_meta)
    
    print("="*100)
    print("====== PREPARING DATA ============")
    print("="*100 + '\n')
    
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
        
    if reduce_cardinality:
        df_input = reduce_cardinality(df_input, 'OCCUP',    700, 999)
        df_input = reduce_cardinality(df_input, 'POB',      500, 'other')
        df_input = reduce_cardinality(df_input, 'ANCSTRY1', 500, 999)
        df_input = reduce_cardinality(df_input, 'ANCSTRY2', 500, 999)
        
    if len(dummy_features) > 0:
        dummy_features = list(set(dummy_features))
        dummy_features = [d for d in dummy_features if d not in [target_feature]]
        
        cardinalities = [len(df_input[d].unique()) for d in dummy_features]
        
        
        df_input = pd.get_dummies(df_input,
                                  columns=dummy_features, 
                                  drop_first=True)
        
        features = df_input.columns
    
    
    cat_features = [c for c in features if c not in NUM_FEATURES]
    df_X, encoders = _prepare_data(df_input, 
                        target      = target_feature, 
                        cols        = features, 
                        cat_cols    = cat_features, 
                        encode_cats = encode_cats,
                        verbose     = verbose,)
    
    print(f"Rows          : {df_X.shape[0]} Features: {df_X.shape[1]}")

    X_train, y_train, X_test, y_test = _split_data(df_X, target=target_feature, scaler=scaler)
    print(f"Training data : {X_train.shape[0]} \nTest data     : {X_test.shape[0]}")
    
    print("="*100)
    print("====== TRAINING MODELS ============")
    print("="*100 + '\n')
    
    results = []
    coefs   = []
    models  = []
    
    best_model = None
    best_model_score = -100000.0
    
    model_obj = {"linearRegression":LinearRegression,
                 "gradientBoostedTree":GradientBoostingRegressor}[model_name]
    for i in range(n_repeats):
        
        model = model_obj(**model_params)
        
        print(fr"{i}/{n_repeats}", end='\r')
        X_train, y_train, X_test, y_test = _split_data(df_X, 
                                                       target=target_feature, 
                                                       scaler=scaler, 
                                                       shuffle=True)
        reg      = model.fit(X_train, y_train)
        coefs   += [reg.coef_]
        results += [_regression_metrics(reg, X_test, y_test, verbose=False)]
        models  += [model]
        
        if results[-1]['score'] > best_model_score:
            best_model = model

    err = [r['mean_abs_err'] for r in results]
    
    print("="*100)
    print("====== TESTING MODELS ============")
    print("="*100 + '\n')
    
    print(f"Error : ${np.mean(err):.2f} +/- {np.std(err):.2f}")
    
    
    xvals_ = [c for c in df_X.columns if c != target_feature]
    yvals_, yvals_err_ = np.array(coefs).mean(axis=0), np.array(coefs).std(axis=0)

    fig, ax = plt.subplots(1,1,figsize=(20,6))

    # yvals_ = reg.coef_
    idx = np.argsort(np.abs(yvals_))[::-1]
    xvals     = np.array([xvals_[i] for i in idx])
    yvals     = np.array([yvals_[i] for i in idx])
    yvals_err = np.array([yvals_err_[i] for i in idx])

    plt.scatter(xvals, yvals - yvals_err, color="black", s=10)
    plt.scatter(xvals, yvals + yvals_err, color="black", s=10)

    plt.plot([0,df_X.shape[1]-1],[0,0], color="gray", ls=":")
    for c, coef in zip(xvals, yvals):
        color = 'g' if coef>0 else 'r'
        plt.plot([c,c],[0,coef], color=color)

    plt.xticks(rotation=90)
    plt.show()

    print('\n')
    print("MOST IMPORTANT FEATURES\n")
    for i in np.where((reg.coef_ > 2000) | (reg.coef_ < -2000))[0]:
        col_name = df_X.drop(target_feature, axis=1).columns[i]
        description = describe_feature(col_name, df_meta)
        print(f"{col_name:<10} {reg.coef_[i]:<10.2f} {description}")


    print('\n')
    print("LEAST IMPORTANT FEATURES\n")
    for i in np.where(abs(reg.coef_) < 50 )[0]:
        col_name = df_X.drop(target_feature, axis=1).columns[i]
        description = describe_feature(col_name, df_meta)
        print(f"{col_name:<10} {reg.coef_[i]:<10.2f} {description}")

    return best_model
        
def train_decision_trees(df, 
                            target_feature,
                            keep_features = [],
                            drop_features = [],
                            model_type = 'hist',
                            model_params={},
                            reduce_card=False,
                            n_repeats=10, 
                            test_split=0.05,
                            encode_cats=True,
                            scale=True,
                            verbose=True,NUM_FEATURES=[], df_meta=None):
    
    df, features = _validate_features(df, target_feature, keep_features, drop_features)
    if verbose:
        print_features(df, NUM_FEATURES, df_meta)
    
    print("="*100)
    print("PREPARING DATA")
    print("="*100 + '\n')
    
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
    
    if reduce_cardinality:
        df = reduce_cardinality(df, 'OCCUP',    700, 999)
        df = reduce_cardinality(df, 'POB',      500, 'other')
        df = reduce_cardinality(df, 'ANCSTRY1', 500, 999)
        df = reduce_cardinality(df, 'ANCSTRY2', 500, 999)
    
    cat_features = [c for c in features if c not in NUM_FEATURES]
    df_X, _ = _prepare_data(df, 
                        target      = target_feature, 
                        cols        = features, 
                        cat_cols    = cat_features, 
                        encode_cats = encode_cats,
                        verbose     = verbose)
    
    print("\n")
    print(f"Rows          : {df_X.shape[0]} Features: {df_X.shape[1]}")

    X_train, y_train, X_test, y_test = _split_data(df_X, target=target_feature, scaler=scaler)
    print(f"Training data : {X_train.shape[0]} \nTest data     : {X_test.shape[0]}")
    
    print("="*100)
    print("TRAINING MODELS")
    print("="*100 + '\n')
    
    results = []
    coefs   = []
    models  = []
    
    if model_type == 'hist':
        model_obj = HistGradientBoostingRegressor
    else:
        model_obj  = GradientBoostingRegressor
    best_model = None
    best_model_score = -100000.0
    
    for i in range(n_repeats):
        model = model_obj(**model_params, verbose=verbose)
        
        print(fr"{i}/{n_repeats}", end='\r')
        X_train, y_train, X_test, y_test = _split_data(df_X, 
                                                       target=target_feature, 
                                                       scaler=scaler, 
                                                       shuffle=True)
        reg      = model.fit(X_train, y_train)
        results += [_regression_metrics(reg, X_test, y_test, verbose=False)]
        models  += [model]
        
        if results[-1]['score'] > best_model_score:
            best_model = model

    print("="*100)
    print("FEATURE IMPORTANCE")
    print("="*100 + '\n')
    
    importance = {}
    for a,b in zip(df_X.drop(target_feature, axis=1).columns, 
                   best_model.feature_importances_):
        importance[a] = b
    run_total=0.0
    for a,b in dict(sorted(importance.items(), key=lambda item: item[1])[::-1]).items():
        desc = describe_feature(a, df_meta)
        run_total += b
        print(f"{a:<10} - {b:<5.3f} - {run_total:<3.2f} - {desc}")
        
    err = [r['mean_abs_err'] for r in results]
    
    print("="*100)
    print("TESTING MODELS")
    print("="*100 + '\n')
    
    print(f"Error : ${np.mean(err):.2f} +/- {np.std(err):.2f}")


    return best_model

def train_neural_network(df, 
                            target_feature,
                            keep_features = [],
                            drop_features = [],
                            model_params={ "random_state":1, "max_iter":300,"learning_rate_init":0.01},
                            n_repeats=10, 
                            test_split=0.05,
                            encode_cats=True,
                            scale=True,
                            verbose=True,
                            NUM_FEATURES=[], df_meta=None):
    
    df, features = _validate_features(df, target_feature, keep_features, drop_features)
    if verbose:
        print_features(df, NUM_FEATURES, df_meta)
    
    print("="*100)
    print("====== PREPARING DATA ============")
    print("="*100 + '\n')
    
    if scale:
        scaler = StandardScaler()
    else:
        scaler = None
    
    cat_features = [c for c in features if c not in NUM_FEATURES]
    df_X, _ = _prepare_data(df, 
                        target      = target_feature, 
                        cols        = features, 
                        cat_cols    = cat_features, 
                        encode_cats = encode_cats,
                        verbose     = verbose)
    print(f"Rows          : {df_X.shape[0]} Features: {df_X.shape[1]}")

    X_train, y_train, X_test, y_test = _split_data(df_X, target=target_feature, scaler=scaler)
    print(f"Training data : {X_train.shape[0]} \nTest data     : {X_test.shape[0]}")
    
    print("="*100)
    print("TRAINING MODELS")
    print("="*100 + '\n')
    
    results = []
    coefs   = []
    models  = []
    
    model_obj  = MLPRegressor
    best_model = None
    best_model_score = -100000.0
    
    for i in range(n_repeats):
        model = model_obj(**model_params, 
                          hidden_layer_sizes=(X_train.shape[1],
                                              X_train.shape[1],
                                              X_train.shape[1]),
                          verbose=verbose)
        
        print(fr"{i}/{n_repeats}", end='\r')
        X_train, y_train, X_test, y_test = _split_data(df_X, 
                                                       target=target_feature, 
                                                       scaler=scaler, 
                                                       shuffle=True)
        for i in range(100):
            model.partial_fit(X_train, y_train)
            if verbose:
                print(f"${np.abs(model.predict(X_test) - y_test).mean():.2f}")
    
        results += [_regression_metrics(model, X_test, y_test, verbose=False)]
        models  += [model]
        
        if results[-1]['score'] > best_model_score:
            best_model = model

    print("======= Feature Importance ========")
    importance = {}
    for a,b in zip(df_X.drop(target_feature, axis=1).columns, 
                   best_model.feature_importances_):
        importance[a] = b
    run_total=0.0
    for a,b in dict(sorted(importance.items(), key=lambda item: item[1])[::-1]).items():
        desc = describe_feature(a, df_meta)
        run_total += b
        print(f"{a:<10} - {b:<5.3f} - {run_total:<3.2f} - {desc}")
        
    err = [r['mean_abs_err'] for r in results]
    
    print("="*100)
    print("====== TESTING MODELS ============")
    print("="*100 + '\n')
    
    print(f"Error : ${np.mean(err):.2f} +/- {np.std(err):.2f}")


    return best_model


def print_features(df, NUM_FEATURES, df_meta):
    print("===== FEATURE LIST ======")
    for x in list(df.columns):
        try:
            vartype = 'NUM' if x in NUM_FEATURES else 'CAT'
            if vartype == 'CAT':
                card = len(df[x].unique())
                print(f"{x:<20} - {vartype} - {card:<4} - {describe_feature(x, df_meta)}")
            else:
                print(f"{x:<20} - {vartype} -      - {describe_feature(x, df_meta)}")
        except:
            print(x)

def describe_feature(x, df_meta): 
    v = df_meta[df_meta['Var'] == x]['Description'].values
    if len(v) == 0:
        return ''
    else:
        return v[0]