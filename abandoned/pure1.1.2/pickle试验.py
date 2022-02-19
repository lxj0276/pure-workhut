import pickle

a={'path1':'22','path2':33,'path3':'pii'}
a_pickle=open('a_save.settings','wb')
pickle.dump(a,a_pickle)
a_pickle.close()

b_file=open('a_save.settings','rb')
b=pickle.load(b_file)


c_file=open('a_save.settings','rb')
c=pickle.load(c_file)

class dog():
    def __init__(self):
        self.__dict__=b

d=dog()