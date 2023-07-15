import pickle 
import re
import numpy as np
import pickle


def split_function(filepath):
    function_list = []
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    flag = -1  

    for line in lines:
        text = line.strip()
        if '/' in text:
            text=text.split('/')
            text=text[0]
        if len(text) > 0 and text != "\n":
            if text.split()[0] == "function" or text.split()[0]=='function()' or text.split()[0] == "constructor":
                function_list.append([text])
                flag += 1
            elif len(function_list) > 0 and ("function" or "constructor" in function_list[flag][0]):
                function_list[flag].append(text)
    return function_list

def find_call_value(file_path):
    all_function_list=split_function(file_path)
    call_value_list=[]
    get_call_value_function_list=[]  
    otherFunction_list=[]
    target='.call.value'

    for i in range(len(all_function_list)):
        count=0
        for j in range(len(all_function_list[i])):
            tmp=all_function_list[i][j]
            if re.search(target,tmp,flags=0):
                count +=1
                call_value_list.append(all_function_list[i])
                break
        if count==0:
            otherFunction_list.append(all_function_list[i])

    for i in range(len(call_value_list)):
        tmplist=call_value_list[i][0].split(' ')
        if len(tmplist)>=2:
            tmp=tmplist[1]
            function_tmp=tmp.split('(')
            function_name=function_tmp[0]
            for j in range(len(otherFunction_list)):
                for k in range(len(otherFunction_list[j])):
                    otherfunction=otherFunction_list[j][k]
                    if re.search(function_name,otherfunction,flags=0):
                        get_call_value_function_list.append(otherFunction_list[j])
                        break
    
    return call_value_list,get_call_value_function_list


def main():
    have_call_value_text_number= np.loadtxt("tool/new_reentrancy_number.txt", delimiter=',')
    
    count_call_value_0=0
    count_call_value_1=0
    count_call_vale_list_0=[]
    

    for i in range(len(have_call_value_text_number)):
        source_file_path='tool/re_clean_reentrancy/'+str(int(have_call_value_text_number[i]))+'.sol'
        save_file_path_pkl='tool/extract_function_reentrancy/'+str(int(have_call_value_text_number[i]))+'.pkl'
        save_file_path_txt='tool/extract_function_reentrancy/'+str(int(have_call_value_text_number[i]))+'.txt'
        
        save_file_path_single_pkl='tool/extract_reentrancy_function_single/'+str(int(have_call_value_text_number[i]))+'.pkl'
        save_file_path_single_txt='tool/extract_reentrancy_function_single/'+str(int(have_call_value_text_number[i]))+'.txt'

        call_value_list_result,get_call_value_function_list_result=find_call_value(source_file_path)

        if len(call_value_list_result)==0 and len(get_call_value_function_list_result)==0:
            count_call_value_0+=1
            count_call_vale_list_0.append(have_call_value_text_number[i])
        
        if len(call_value_list_result)==0:
            count_call_value_1+=1
        else:
            count_call_value_0+=1

        new_call_value=np.asarray(call_value_list_result+get_call_value_function_list_result)
        np.savetxt(save_file_path_txt,new_call_value,fmt='%s')
 
        with open(save_file_path_pkl,'wb') as f:
            pickle.dump(call_value_list_result+get_call_value_function_list_result,f)
        
        with open(save_file_path_single_pkl,'wb') as f:
            pickle.dump(call_value_list_result,f)

        new_call_value_single=np.asarray(call_value_list_result)
        np.savetxt(save_file_path_single_txt,new_call_value_single,fmt='%s')

if __name__ == "__main__":
    main()



