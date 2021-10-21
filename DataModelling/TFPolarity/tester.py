from modelWrapper import modelWrapper
import pickle
import os

def main():
    model = modelWrapper(name='TF_MODEL')
    #current model only does labels 0 and 4 and has limited vocab
    #0 = negative, 4 = positive
    output_dict = model.inference("i am very angry")
    print(output_dict)

main()
