class Student:

    def __init__(self,n,a):
        self.full_name = n
        self.age = a

    def get_age(self):
        return self.age

class AIStudent(Student):

    def __init__(self,n,a,s):
        super(AIStudent,self).__init__(n,a)
        self.section_num = s

    def get_age(self):
        print("Age: " + str(self.age))

if __name__ == '__main__':


    links = tuple()
    print(links)
    lista = list(links)
    lista.append((1,2))
    links = tuple(lista)
    print(links)