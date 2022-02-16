import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm 
import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from natasha import Segmenter, Doc
from pullenti_wrapper.processor import (
    Processor,
    GEO,
    DATE,
    ORGANIZATION,
    PERSON
)
 
import warnings
warnings.filterwarnings('ignore')
 
NOT_MENTIONED = 'отсутствует информация/информация не найдена'
POROG = 0.95
 
 
def cosine(u, v):
    '''
    находит косинусное сходство между векторами
    '''
    return np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v.T))
 
def select_latest(names):
    '''
    выбор последней версии из документов с несколькими версиями
    '''
    res = defaultdict(tuple)
    
    for name in names:
        numbers = re.findall(r'\d+', name)
        res[name] = (sum(map(int, numbers)), len(name))
        
    return sorted(res.items(), key=lambda x: (x[1]), reverse=True)[0][0]
 
def selecting_files(f_list, m): # будет время - переработать на поиск циферок по регуляркам 
    '''
    находит разные версии одного файла
    '''
    
    files_list = []
    fl_dict = {n:i for n,i in enumerate(f_list)}
    fl_vec = {n:i for n,i in enumerate(m.encode(f_list))}
    
    while len(fl_vec)>0:
        combo = fl_vec.popitem()
        file_vec = combo[1]
        similar_names = [fl_dict.pop(combo[0])]
        
        for i in list(fl_vec.keys()):
            sim = cosine(file_vec, fl_vec[i])
            
            if sim >= 0.98:
                del fl_vec[i]
                similar_names.append(fl_dict.pop(i))
                
        name = select_latest(similar_names)
        files_list.append(name)
        
    return files_list
 
 
def choose_files(model, dir_base):
    '''
    формируем дф с названием файлов, которые будем рассматривать
    
    Parameters:
        model (): модель
        dir_base (): каталог для текстовых файлов
        
    Returns:
        df (list): list с названием файлов
    '''
    dir_base = 'data/'
    df = pd.DataFrame(columns = ['filename', 'comitet_num','comitet', 'path2file'])
 
 
    for directory in tqdm(os.listdir(dir_base)):
        files_in_directory = os.listdir(dir_base + directory)
 
        if '.ipynb_checkpoints' in files_in_directory: files_in_directory.remove('.ipynb_checkpoints')  # временно!
 
        tmp_df = pd.DataFrame(selecting_files(files_in_directory, model),columns = ['filename'])
        tmp_df['comitet_num'] = tmp_df['filename'].apply(lambda x: x[:3])
        tmp_df['comitet'] = directory.split('_')[-1]
        tmp_df['file_path'] = dir_base + directory + '/' + tmp_df['filename']
        df = df.append(tmp_df)
    
    df = df.sort_values(['comitet_num', 'comitet'], ascending = [True, False]).reset_index(drop = True)
    
    return df
 
 
def check_assignment_exist(inspect_assignment, assignment_df, filename):
    '''
    Проверяет поручение на существование в дф (есть ли уже запись поручения)
    
    Parameters:
        inspect_assignment (list): список информации о поручении
        assignment_df (df): дф с поручениями
    
    Returns:
        flg_exist (bool): если поручение есть в дф поручений - 1, иначе 0
        assignment_df (df): дф поручений
    '''
    sent_comitet_num = filename.split('/')[-1][:3]
    flg_exist = 0
    
    for assignment_num,existing_assignment in enumerate(assignment_df['a_vec']):
        cosine_value = cosine(inspect_assignment[-1], existing_assignment) > 0.99
        
        if cosine_value and sent_comitet_num == assignment_df.loc[assignment_num, 'comitet_num']:
            flg_exist = 1
            # проверка сформулировано
            if inspect_assignment[6] == 'Да' and assignment_df.loc[assignment_num, 'is_formulated'] == 'Нет':
                assignment_df.loc[assignment_num, 'is_formulated'] = 'Да'
            
            # проверка поставлено на контроль
            if inspect_assignment[7] == 'Да' and assignment_df.loc[assignment_num, 'is_stated'] == 'Нет':
                assignment_df.loc[assignment_num, 'filepath'] = inspect_assignment[1]
                assignment_df.loc[assignment_num, 'is_stated'] == 'Да'
        elif cosine_value and sent_comitet_num > assignment_df.loc[assignment_num, 'comitet_num']:
            assignment_df.loc[assignment_num, 'dbl_count'] += 1
            assignment_df.loc[assignment_num, 'dbl_assignment'].append(filename)
            
    return flg_exist, assignment_df
 
def compare_to_assignment(sent, filename, assignment_df, model, pattern, ssilka, porog):
    '''
    сравнивает предложение с найденными поручениями
    
    Parameters:
        sent(str): предложение
        filename(str): название файла предложения
        assignment_df(df): дф с поручениями
        pattern (): шаблон регулярки для отбора символов
        porog(float): threshold
    
    Returns:
        assignment_df(df): измененный дф поручений(если предложение является упоминанем поручения)
    '''
    sent_comitet_num = filename.split('/')[-1][:3]
    
    for assignment_num, assignment_vec in enumerate(assignment_df['a_vec']):
        comitet_num = assignment_df.loc[assignment_num, 'comitet_num']
        if (str(comitet_num) in ssilka) and (sent_comitet_num > comitet_num) and (assignment_df.loc[assignment_num, 'is_stated'] == 'Да'):
            sent_vec = model.encode([' '.join(pattern.findall(sent))])
            cosine_value = cosine(sent_vec, assignment_vec)
 
            if porog < cosine_value < 0.99:
                    if assignment_df.loc[assignment_num, 'mention'] == NOT_MENTIONED:
                        assignment_df.loc[assignment_num, 'mention'] = 1
                        assignment_df.loc[assignment_num, 'mention_file'] = [filename]
                        assignment_df.loc[assignment_num, 'mention_sent'] = [sent]
                    else:
                        assignment_df.loc[assignment_num, 'mention'] += 1
                        assignment_df.loc[assignment_num, 'mention_file'].append(filename)
                        assignment_df.loc[assignment_num, 'mention_sent'].append(sent)
 
    return assignment_df
 
 
 
 
def kinda_main_func(df, model, processor):
    '''
    итерируемся по тексам, далее итерируемся по предложениям. Если это поручение,
    то записываем в поручения, если нет, то сравниваем с поручениями
    
    Parameters:
        filenames (list): list с названиями файлов
        model (): моделька для векторизации
    
    Returns:
        -
    
    Output:
        xlsx с поручениями
    '''
    
    columns_list =['assignment','filepath', 'comitet','comitet_num','till_date','responsible_person',
               'is_formulated' ,'is_stated', 'dbl_count', 'dbl_assignment', 'mention', 'mention_file', 'mention_sent', 'a_vec']
    pattern_assignment = re.compile(r'[Пп]оручить|[Вв]ынести на рассмотрение')
    pattern_k = re.compile(r'[А-Яа-я]+')
    pattern_date = re.compile(r'\d{2}.\d{2}.\d{4}|\d{2}.{0,2}[А-Яа-я]+.{0,2}\d{4}')
    #pattern_fio = re.compile(r'\(\s*[А-Я]?\.?\s*[А-Я]?\.\s*[А-Яа-яЁё]+.\)|\(\s*[А-Яа-яЁё]+\s*[А-Я]\.\s*[А-Я]?\.?\s*\)')
    pattern_kpp = re.compile(r'КРР.{0,3}№.{0,3}\d{3}|[Кк]омитет.{0,3}по рыночным рискам.{0,3}№.?\d{3}')
    signed = re.compile('ДОКУМЕНТ УТВЕРЖДЕН  УСИЛЕННОЙ КВАЛИФИЦИРОВАННОЙ  ЭЛЕКТРОННОЙ ПОДПИСЬЮ')
 
    start_time = time.time()
    assignment_df = pd.DataFrame(columns=columns_list)
    pos = 0
    seg = Segmenter()
    dbl_assignments_counter = 0
 
    for filename in tqdm(df['file_path']):  # по порядку берем названия файлов
        with open(filename) as f:  # открываем 
            text = f.read()
 
        flg_is_solution = 1 if len(re.findall('[Рр]ешени[ея]', filename)) > 0 else 0  # файл является "решением"
        flg_is_project_solution = 1 if len(re.findall('[Пп]роект.+[Рр]ешени[ея]', filename)) > 0 else 0  # файл является "проект решения"
        text = text.replace('\n', ' ').replace(';', ' ')#.split('  ')
 
        # сегментация на предложения
        found_commitets = pattern_kpp.findall(text)
        ssilka = re.findall('\d+',(' '.join(found_commitets)))
        doc = Doc(text)
        doc.segment(seg)
        texts = list(map(lambda x: x.text, doc.sents))
 
        if len(texts) > 1:
            flg_is_signed = 1 if len(signed.findall(text)) > 0 else 0
        else:
            flg_is_signed = 0
 
        for sent_pos, sent in enumerate(texts):  # берем по одному предложению из текста
            sent_is_assignment = 0
 
            if len(pattern_k.findall(' '.join(pattern_k.findall(sent)))) > 4:  # исключаем нереальные предложения
                if len(pattern_assignment.findall(sent)) != 0:  # если есть кл.слова поручений - проверяем на то что это поручение
                    prt = processor(sent)
                    NER_responsible = []
 
                    for i in range(len(prt.matches)):
                        NER_responsible.extend([value for key,value in prt.matches[i].referent.slots if key in ['LASTNAME']])
 
                    new_row = [
                        sent,  # текст предложения 
                        filename,  # путь к файлу этого поручения
                        filename.split('/')[1].split('_')[-1],  # достаем комитет (альфа, бета, сигма)
                        filename.split('/')[-1][:3],  # номер заседания
                        ' '.join(pattern_date.findall(sent)),  # дата в тексте (считается, что это контрольная дата для поручения)
                        NER_responsible,  # фио ответственного
                        'Да' if flg_is_project_solution else 'Нет',  # сформулированное поручение 
                        'Да' if (flg_is_solution and not flg_is_project_solution) or flg_is_signed else 'Нет',  # поставлено на контроль
                        0,  # счетчик упоминаний данного поручения
                        [],
                        NOT_MENTIONED,  # упоминание
                        NOT_MENTIONED,  # файл упоминания 
                        NOT_MENTIONED   # предложения упоминания
                    ]
 
                    sent_is_assignment = 1
                    new_row.append(model.encode([' '.join(pattern_k.findall(sent))]))  # вектор поручения
                    exist_assignment, assignment_df = check_assignment_exist(new_row, assignment_df, filename)  # проверяем,записали ли мы такое поручение 
 
                    if exist_assignment == 0:  # если нет, то добавляем как новое, иначе на предыдущем шаге были изменения в датафрейме
                        assignment_df.loc[pos] = new_row
                        pos+=1
 
                if len(ssilka) > 0 and not sent_is_assignment and len(assignment_df) != 0:
                    assignment_df = compare_to_assignment(sent, filename, assignment_df, model, pattern_k, ssilka, POROG)
 
    assignment_df['till_date'] = assignment_df['till_date'].apply(lambda x: str(x).replace('[','').replace(']',''))
    assignment_df['responsible_person'] = assignment_df['responsible_person'].apply(lambda x: str(x).replace('[','').replace(']','').replace('(','').replace(')',''))
    assignment_df.drop(['a_vec'], axis = 1).to_excel('output_file.xlsx', encoding = 'cp1251')
    
    
def classic(dir_base='data/'):
    start_time = time.time()
    model = SentenceTransformer('rubert_base_cased_sentence/')
    processor = Processor([PERSON])
    df = choose_files(model, dir_base)
    kinda_main_func(df, model, processor)
    time_delta = time.time() - start_time
 
    
if __name__ == '__main__':
    classic('res_raw_files')
