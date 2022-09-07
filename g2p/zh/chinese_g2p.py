from pypinyin import pinyin, lazy_pinyin, Style

def pinyin_list_to_str(res:list, with_space=True):
    string = ''
    for i in res:
        if type(i) == list:
            for j in i:
                string+= j
                if with_space:
                    string += ' '
        else:
            string+= i
            if with_space:
                string += ' '
        
    return string

def g2p(text, mode, with_space=True):
    if mode == 'normal':
        return pinyin_list_to_str(pinyin(text), with_space) 
    elif mode == 'tone3':
        return pinyin_list_to_str(pinyin(text, style=Style.TONE3), with_space)
    elif mode == 'bopomofo':
        return pinyin_list_to_str(pinyin(text, style=Style.BOPOMOFO), with_space)

if __name__ == '__main__':
    [print(g2p('杏铃铃，呀呀呀！', mode)) for mode in ['normal', 'tone3', 'bopomofo']]