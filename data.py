import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter as ct

def personal_data_analyze(game_info: pd.DataFrame = None) -> (str):
    # parameters
    Balltype_lists_A = []
    Balltype_lists_B = []
    BreakPoint = 0
    # data 2 list
    df_list = game_info.values.tolist()
    # separate A B
    for i in range(len(df_list) - 1):
        Hitter = df_list[i + 1][2]
        Balltype= df_list[i + 1][12]
        if Hitter == 'A':
            Balltype_lists_A.append(Balltype)
        else:
            Balltype_lists_B.append(Balltype)

    counterA = ct(Balltype_lists_A)
    counterB = ct(Balltype_lists_B)

    # Get Server, Winner
    ServerList = game_info[game_info['ShotSeq'] == 1]
    WinnerList = game_info[game_info['Winner'].isin(['A', 'B'])]
    Server = ServerList['Hitter'].tolist()
    Winner = WinnerList['Winner'].tolist()
    WinnerCT = ct(Winner)

    # Win ratios
    WinRate = round(WinnerCT['A']/len(ServerList)*100, 2)

    # BreakPoint ratios
    for i in range(len(ServerList)):
        if Server[i] == Winner[i]:
            BreakPoint += 1
    BP_Rate = round(BreakPoint/len(ServerList)*100, 2)

    ret_str = f'The HitterA win rate is {WinRate}%, BreakPoint rate is {BP_Rate}%'

    # labels Balltype
    labels = ['long', 'flat', 'smash', 'net shot', 'slice', 'lofted', 'push']
    sizeA = [counterA[3], counterA[4], counterA[5], counterA[6], counterA[7], counterA[8], counterA[9]]
    sizeB = [counterB[3], counterB[4], counterB[5], counterB[6], counterB[7], counterB[8], counterB[9]]

    # filter 0 count index.
    filtered_sizeA = [size for size in sizeA if size != 0]
    filtered_labelsA = [label for label, size in zip(labels, sizeA) if size != 0]
    filtered_sizeB = [size for size in sizeB if size != 0]
    filtered_labelsB = [label for label, size in zip(labels, sizeB) if size != 0]

    plt.figure()
    fig, ax = plt.subplots()
    ax.pie(filtered_sizeA, labels=filtered_labelsA, autopct='%1.1f%%')
    plt.savefig('HitterA_balltype.png')

    plt.figure()
    fig, ax = plt.subplots()
    ax.pie(filtered_sizeB, labels=filtered_labelsB, autopct='%1.1f%%')
    plt.savefig('HitterB_balltype.png')
    
    return ret_str

def data_analyze(game_info: pd.DataFrame = None) -> (str):
    # parameters
    index_mapping = {3: 'long', 4: 'flat', 5: 'smash', 6: 'net shot', 7: 'slice', 8: 'lofted', 9: 'push'}
    RoundHead_value_A = Backhand_value_A = Ballheight_value_A = 0
    RoundHead_value_B = Backhand_value_B = Ballheight_value_B = 0
    Balltype_lists_A = []
    Balltype_lists_B = []
    # data to list
    df_list = game_info.values.tolist()
    if df_list[0][12] == 1:
        service:str = 'short service'
    else:
        service:str = 'high service'
    
    # separate A B data
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
    # analyze A data
    RH_ratio_A = 1 - (HitterA_data[0] - HitterA_data[3]) / HitterA_data[3]
    BH_ratio_A = 1 - (HitterA_data[1] - HitterA_data[3]) / HitterA_data[3]
    if HitterA_data[2]/HitterA_data[3] > 1.5:
        BallHA:str = "low"
    else:
        BallHA:str = "high"
    # analyze B data
    RH_ratio_B = 1 - (HitterB_data[0] - HitterB_data[3]) / HitterB_data[3]
    BH_ratio_B = 1 - (HitterB_data[1] - HitterB_data[3]) / HitterB_data[3]
    if HitterB_data[2]/HitterB_data[3] > 1.5:
        BallHB:str = "low"
    else:
        BallHB:str = "high"
    # initiative
    if BH_ratio_A >  BH_ratio_B:
        initiative:str = 'Hitter B.'
    else:
        initiative:str = 'Hitter A.'
    # get max mapping value
    counterA = ct(Balltype_lists_A)
    counterB = ct(Balltype_lists_B)
    max_keys = range(3, 10)
    maxA = max([counterA[key] for key in max_keys])
    maxB = max([counterB[key] for key in max_keys])
    maxY = 0
    if maxA > maxB:
        maxY = maxA
    else:
        maxY = maxB

    # plot A fig
    data_replaced_A = [index_mapping[num] if num in index_mapping else num for num in Balltype_lists_A]
    counter_A = ct(data_replaced_A)
    keys_A = list(index_mapping.values())
    values_A = [counter_A[key] for key in keys_A]

    plt.figure()
    plt.bar(keys_A, values_A, color='skyblue', alpha=0.7)
    plt.xlabel('Balltype')
    plt.ylabel('Frequency')
    plt.title('Balltype histogram')
    plt.ylim(0, maxY)
    plt.show()
    plt.savefig('HitterA.png')

    # plot B fig
    data_replaced_B = [index_mapping[num] if num in index_mapping else num for num in Balltype_lists_B]
    counter_B = ct(data_replaced_B)
    keys_B = list(index_mapping.values())
    values_B = [counter_B[key] for key in keys_B]

    plt.figure()
    plt.bar(keys_B, values_B, color='skyblue', alpha=0.7)
    plt.xlabel('Balltype')
    plt.ylabel('Frequency')
    plt.title('Balltype histogram')
    plt.ylim(0, maxY)
    plt.show()
    plt.savefig('HitterB.png')

    Match_info:str = f"In this match, serving side is Hitter {df_list[0][2]}, hitting a {service}.\n"
    Winner_info:str = f"The winner is Hitter {df_list[len(df_list)-1][13]}.\n"
    HitterA_info:str = f"Hitter A, RoundHead ratio is {round(RH_ratio_A, 2)}%, BackhandRatio is {round(BH_ratio_A, 2)}%, Average hit on {BallHA} Ballheight.\n"
    HitterB_info:str = f"Hitter B, RoundHead ratio is {round(RH_ratio_B, 2)}%, BackhandRatio is {round(BH_ratio_B, 2)}%, Average hit on {BallHB} Ballheight.\n"
    Initiative_info:str = f"{initiative} take the initiative."

    ret_str:str = Match_info + Winner_info + HitterA_info +HitterB_info + Initiative_info
    
    return ret_str

if __name__ == '__main__':
    # string = data_analyze(pd.read_csv("00001_S2.csv"))
    # print(string)
    string = personal_data_analyze(pd.read_csv("00001_S2.csv"))
    print(string)