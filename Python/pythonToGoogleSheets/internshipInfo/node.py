class Node:
    def __init__(self, company:str, loc:str, title:str, length:str, start:str, pay:str, requiredSkills:str, link:str):
        self.company = company
        self.loc = loc
        self.title = title
        self.length = length
        self.start = start
        self.pay = pay
        self.requiredSkills = requiredSkills
        self.link = link
        self.next:Node = None
        self.prev:Node = None
        self.rank:int = None


