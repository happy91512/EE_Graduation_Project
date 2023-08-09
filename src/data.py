import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter as ct

index_mapping = {3: 'long', 4: 'flat', 5: 'smash', 6: 'net shot', 7: 'slice', 8: 'lofted', 9: 'push'}
# csv to list
def read_csv2list(df):
    df_list = df.values.tolist()
    return df_list

def initial_analyze(df_list):
    if df_list[0][12] == 1:
        service:str = 'short service'
    else:
        service:str = 'high service'
    return(f"In this match, serving side is Hitter {df_list[0][2]}, hitting a {service}."), (f"The winner is Hitter {df_list[len(df_list)-1][13]}.")

# divide A B data
def separateABdata(df_list):
    RoundHead_value_A = Backhand_value_A = Ballheight_value_A = 0
    RoundHead_value_B = Backhand_value_B = Ballheight_value_B = 0
    Balltype_lists_A = []
    Balltype_lists_B = []
    for i in range(len(df_list) - 1):
        Hitter, RoundHead, Backhand, Ballheight = df_list[i + 1][2:6]
        Balltype= df_list[i + 1][12]
        if Hitter == 'A':
            RoundHead_value_A = RoundHead_value_A + RoundHead
            Backhand_value_A  = Backhand_value_A + Backhand
            Ballheight_value_A = Ballheight_value_A + Ballheight
            Balltype_lists_A.append(Balltype)
        else:
            RoundHead_value_B = RoundHead_value_B + RoundHead
            Backhand_value_B  = Backhand_value_B + Backhand
            Ballheight_value_B = Ballheight_value_B + Ballheight
            Balltype_lists_B.append(Balltype)

    HitterA_data = [RoundHead_value_A, Backhand_value_A, Ballheight_value_A, len(Balltype_lists_A)]
    HitterB_data = [RoundHead_value_B, Backhand_value_B, Ballheight_value_B, len(Balltype_lists_B)]
    return HitterA_data, Balltype_lists_A, HitterB_data, Balltype_lists_B

def dataAnalyze_A(A_data):
    RH_ratio = 1 - (A_data[0] - A_data[3]) / A_data[3]
    BH_ratio = 1 - (A_data[1] - A_data[3]) / A_data[3]
    if A_data[2]/A_data[3] > 1.5:
        BallH:str = "low"
    else:
        BallH:str = "high"
    return(f"Hitter A, RoundHead ratio is {round(RH_ratio, 2)}%, BackhandRatio is {round(BH_ratio, 2)}%, {BallH} Ballheight ")

def dataAnalyze_B(B_data):
    RH_ratio = 1 - (B_data[0] - B_data[3]) / B_data[3]
    BH_ratio = 1 - (B_data[1] - B_data[3]) / B_data[3]
    if B_data[2]/B_data[3] > 1.5:
        BallH:str = "low"
    else:
        BallH:str = "high"
    return(f"Hitter B, RoundHead ratio is {round(RH_ratio, 2)}%, BackhandRatio is {round(BH_ratio, 2)}%, {BallH} Ballheight ")

def get_maxYvalue(BT_A, BT_B):
    counterA = ct(BT_A)
    counterB = ct(BT_B)
    keys = range(3, 10)
    maxA = max([counterA[key] for key in keys])
    maxB = max([counterB[key] for key in keys])
    maxY = 0
    if maxA > maxB:
        maxY = maxA
    else:
        maxY = maxB
    return maxY

def Data_VisualizationA(BT_A, maxY):
    data_replaced_A = [index_mapping[num] if num in index_mapping else num for num in BT_A]
    counter = ct(data_replaced_A)
    keys = list(index_mapping.values())
    values = [counter[key] for key in keys]
    A = plt.figure(dpi=400)
    plt.bar(keys, values, color='skyblue', alpha=0.7)
    plt.xlabel('Balltype')
    plt.ylabel('Frequency')
    plt.title('Balltype histogram - A')
    plt.ylim(0, maxY)
    return A
    # plt.savefig('HitterA.png')
    
    

def Data_VisualizationB(BT_B, maxY):
    index_mapping = {3: 'long', 4: 'flat', 5: 'smash', 6: 'net shot', 7: 'slice', 8: 'lofted', 9: 'push'}
    data_replaced_B = [index_mapping[num] if num in index_mapping else num for num in BT_B]
    counter = ct(data_replaced_B)
    keys = list(index_mapping.values())
    values = [counter[key] for key in keys]
    B = plt.figure(dpi=400)
    plt.bar(keys, values, color='skyblue', alpha=0.7)
    plt.xlabel('Balltype')
    plt.ylabel('Frequency')
    plt.title('Balltype histogram - B')
    plt.ylim(0, maxY)
    return B
    # plt.savefig('HitterB.png')
    
    

def analyze(df:pd.DataFrame):
    csv = read_csv2list(df)
    str1, str2 = initial_analyze(csv)
    HitterA_data, Balltype_lists_A, HitterB_data, Balltype_lists_B = separateABdata(csv)
    str3 = dataAnalyze_A(HitterA_data)
    str4 = dataAnalyze_B(HitterB_data)
    maxY = get_maxYvalue(Balltype_lists_A, Balltype_lists_B)
    figA = Data_VisualizationA(Balltype_lists_A, maxY)
    figB = Data_VisualizationB(Balltype_lists_B, maxY)
    str_list = str1 + '\n' + str2+ '\n' + str3 + '\n' + str4
    return str_list, figA, figB
    

if __name__ == '__main__':
    df = pd.read_csv("saved_csv/00001_S2.csv")
    csv = read_csv2list(df)
    initial_analyze(csv)
    HitterA_data, Balltype_lists_A, HitterB_data, Balltype_lists_B = separateABdata(csv)
    dataAnalyze_A(HitterA_data)
    dataAnalyze_B(HitterB_data)
    maxY = get_maxYvalue(Balltype_lists_A, Balltype_lists_B)
    Data_VisualizationA(Balltype_lists_A, maxY)
    Data_VisualizationB(Balltype_lists_B, maxY)