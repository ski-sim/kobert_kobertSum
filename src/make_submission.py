import json
import numpy as np
import pandas as pd
import time
import re
import sys
import os

PROBLEM = 'ext'

## 사용할 path 정의
# PROJECT_DIR = '/home/uoneway/Project/PreSumm_ko'
PROJECT_DIR = '..'
print(PROJECT_DIR)

DATA_DIR = f'{PROJECT_DIR}/{PROBLEM}/data'
RAW_DATA_DIR = DATA_DIR + '/raw'
JSON_DATA_DIR = DATA_DIR + '/json_data'
BERT_DATA_DIR = DATA_DIR + '/bert_data' 
LOG_DIR = f'{PROJECT_DIR}/{PROBLEM}/logs'
LOG_PREPO_FILE = LOG_DIR + '/preprocessing.log' 

MODEL_DIR = f'{PROJECT_DIR}/{PROBLEM}/models' 
RESULT_DIR = f'{PROJECT_DIR}/{PROBLEM}/results' 

# python make_submission.py result_1209_1236_step_7000.candidate
if __name__ == '__main__':
    # test set
    with open(RAW_DATA_DIR + '/test.jsonl', 'r',encoding='utf-8') as json_file:
        json_list = list(json_file)

    tests = []
    for json_str in json_list:
        line = json.loads(json_str)
        tests.append(line)
    test_df = pd.DataFrame(tests)

    # 추론결과
    with open(RESULT_DIR + '/' + sys.argv[1], 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # print(lines)
    test_pred_list = []
    for line in lines:
        sum_sents_text, sum_sents_idxes = line.rsplit(r'[', maxsplit=1)
        sum_sents_text = sum_sents_text.replace('<q>', '\n')
        sum_sents_idx_list = [ int(str.strip(i)) for i in sum_sents_idxes[:-2].split(', ')]
        test_pred_list.append({'sum_sents_tokenized': sum_sents_text, 
                            'sum_sents_idxes': sum_sents_idx_list
                            })
    result_df = pd.merge(test_df, pd.DataFrame(test_pred_list), how="left", left_index=True, right_index=True)
    def safe_summary(row):
        
        original = np.array(row['article_original'])
        print('index::::',original)
        indices = row['sum_sents_idxes']

        # indices가 리스트인지 확인하고, 그렇지 않으면 적절하게 처리
        if not isinstance(indices, list):
            if pd.isna(indices):  # NaN 처리
                return "No indices provided"
            indices = [int(indices)]  # 숫자를 리스트로 변환

        if all(idx < len(original) for idx in indices):  # 모든 인덱스 검증
            return '\n'.join(list(original[indices]))
        else:
            return "Invalid index"  # 오류 메시지 반환 또는 다른 처리

    result_df['summary'] = result_df.apply(safe_summary, axis=1)


    #result_df['summary'] = result_df.apply(lambda row: '\n'.join(list(np.array(row['article_original'])[row['sum_sents_idxes']])) , axis=1)
    #result_df['summary'] = result_df.apply(lambda row: '\n'.join(list(np.array(row['article_original'])[list(filter(lambda x: x is not None, row['sum_sents_idxes']))])) , axis=1)
    print(result_df['sum_sents_idxes'])
    
    result_df.to_csv(RAW_DATA_DIR + '/extractive_sample_submission_v2.csv')
    
    submit_df = pd.read_csv(RAW_DATA_DIR + '/extractive_sample_submission_v2.csv')
    submit_df.drop(['summary'], axis=1, inplace=True)

    #print(result_df['id'].dtypes)
    #print(submit_df.dtypes)

    result_df['id'] = result_df['id'].astype(int)
    #print(result_df['id'].dtypes)

    submit_df  = pd.merge(submit_df, result_df.loc[:, ['id', 'summary']], how="left", left_on="id", right_on="id")
    #print(submit_df.isnull().sum())

    ## 결과 통계치 보기
    # word
    abstractive_word_counts = submit_df['summary'].apply(lambda x:len(re.split('\s', x)))
    #print(abstractive_word_counts.describe())

    # export
    now = time.strftime('%y%m%d_%H%M')
    submit_df.to_csv(f'{RESULT_DIR}/submission_{now}.csv', index=False, encoding="utf-8-sig")