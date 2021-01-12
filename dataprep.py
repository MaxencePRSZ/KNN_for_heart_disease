import pandas as pd


def dataprep():
    df = pd.read_csv("./dataset/heart.csv")
    df.drop('fbs', inplace=True, axis=1)
    df.drop('chol', inplace=True, axis=1)
    df.drop('restecg', inplace=True, axis=1)
    df.drop('trestbps', inplace=True, axis=1)
    df.drop('age', inplace=True, axis=1)
    df.drop('sex', inplace=True, axis=1)
    df.drop('thal', inplace=True, axis=1)
    df.drop('slope', inplace=True, axis=1)
    df = pd.get_dummies(df, columns=['cp'], prefix = 'cp')
    df = pd.get_dummies(df, columns=['ca'], prefix = 'ca')
    columns = df.columns.to_list()
    columns.append(columns.pop(3))
    df = df[columns]
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df.values.tolist()


def print_corr_matr(corr_matrices):
    tot_acc, tot_predic_pos, tot_recall_pos = 0, 0, 0
    tot_fsc_pos, tot_predic_neg, tot_recall_neg = 0, 0, 0
    tot_fsc_neg, tot_fsc = 0, 0
    
    final_matrix = [[0,0],[0,0]]

    header = "fold n° %d \n"
    core = "   TP    FP   \n   %d    %d   \n          \n   FN    TN   \n   %d    %d   \n"
    footer1 = "Accuracy : %.3f | Pos predic : %.3f | Pos recall : %.3f"
    footer2 = "f-score pos : %.3f | Neg predic : %.3f | Neg recall : %.3f"
    footer3 = "f-score neg : %.3f | f-score : %.3f\n"
              
    for i, matrix in enumerate(corr_matrices):
        final_matrix[0][0] += matrix[0][0]
        final_matrix[0][1] += matrix[0][1]
        final_matrix[1][0] += matrix[1][0]
        final_matrix[1][1] += matrix[1][1]
        print(header % (1+i))
        print(core % (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))
        acc = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
        predic_pos = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        recall_pos = matrix[0][0] / (matrix[0][0] + matrix[1][0])
        print(footer1 % (acc, predic_pos, recall_pos))
        fsc_pos = 2 / ((1 / predic_pos) + (1 / recall_pos))
        predic_neg = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        recall_neg = matrix[1][1] / (matrix[1][1] + matrix[0][1])
        print(footer2 % (fsc_pos, predic_neg, recall_neg))
        fsc_neg = 2 / ((1 / predic_neg) + (1 / recall_neg))
        fsc = (fsc_pos + fsc_neg) / 2
        print(footer3 % (fsc_neg, fsc))

#        tot_acc += acc
#        tot_fsc += fsc
#        tot_fsc_neg += fsc_neg
#        tot_fsc_pos += fsc_pos
#        tot_predic_neg += predic_neg
#        tot_recall_neg += recall_neg
#        tot_predic_pos += predic_pos
#        tot_recall_pos += recall_pos

    print("All folds")
    print(core % (final_matrix[0][0], final_matrix[0][1], final_matrix[1][0], final_matrix[1][1]))
    acc = (final_matrix[0][0] + final_matrix[1][1]) / (final_matrix[0][0] + final_matrix[0][1] + final_matrix[1][0] + final_matrix[1][1])
    predic_pos = final_matrix[0][0] / (final_matrix[0][0] + final_matrix[0][1])
    recall_pos = final_matrix[0][0] / (final_matrix[0][0] + final_matrix[1][0])
    print(footer1 % (acc, predic_pos, recall_pos))
    fsc_pos = 2 / ((1 / predic_pos) + (1 / recall_pos))
    predic_neg = final_matrix[1][1] / (final_matrix[1][1] + final_matrix[1][0])
    recall_neg = final_matrix[1][1] / (final_matrix[1][1] + final_matrix[0][1])
    print(footer2 % (fsc_pos, predic_neg, recall_neg))
    fsc_neg = 2 / ((1 / predic_neg) + (1 / recall_neg))
    fsc = (fsc_pos + fsc_neg) / 2
    print(footer3 % (fsc_neg, fsc))
    AUC = (recall_pos + recall_neg) / 2
    print("AUC : %.7f" % AUC)
