from node import Node

class DoublyLinkedList:
    def __init__(self, firstNode:Node):
        self.head = firstNode
        self.tail = firstNode
        firstNode.rank = 1

    def printLL(self) -> None:
        currentNode = self.head
        #while there is a next node print all the info and set currentNode to the next node
        while(currentNode):
            print(f"Rank: {currentNode.rank}, Company: {currentNode.company}, Location: {currentNode.loc}, Title: {currentNode.title}, Length: {currentNode.length}, Start Date: {currentNode.start}, Pay: {currentNode.pay}, Required Skills: {currentNode.requiredSkills}")
            currentNode = currentNode.next

    def updateRanks(self) -> None:
        #parse through nodes, each time assigning the rank value to the rank of the prev node + 1
        currentNode = self.head
        currentNode = currentNode.next
        if (currentNode != self.head):
            while(currentNode):
                currentNode.rank = currentNode.prev.rank + 1
                currentNode = currentNode.next


    def addNewInternship(self) -> None:
        #gather info about internship
        print("For questions that you do not know answer to just put a ?.")
        newComp = input("What is the name of the company?: ")
        newLoc = input("Where is the internship located?: ")
        newTitle = input("What is the title of the position?: ")
        newLen = input("What is the length on the internship?: ")
        newStart = input("What is the start date?: ")
        newPay = input("What is the pay?: ")
        newReq = input("What are the required skills? Please use comma to separate skills: ")
        newLink = input("What is the link for the job posting?: ")
        #make new node with info from above
        newNode = Node(newComp, newLoc, newTitle, newLen, newStart, newPay, newReq, newLink)
        #print all the other jobs with ranks and ask where they want to put new job
        self.printLL()
        print("If you want to put node last type: Last")
        newRankStr = input("What rank would you like to give it? (Note: it will put the job above the job currently at that rank): ")
        #if they want to put it last
        if (newRankStr.lower() == "last"):
            #assign new node as the tail and set newNode.prev to previous tail and set previous tail.next as new node
            tempNode = self.tail
            self.tail = newNode
            newNode.prev = tempNode
            tempNode.next = newNode
            self.updateRanks()
        #if they want to put it first
        elif(newRankStr == "1"):
            #convert new rank to int
            newRank = int(newRankStr)
            #assign newNode as head
            tempNode = self.head
            self.head = newNode
            #set newNode.next as previous head and previous head.prev as newNode
            newNode.next = tempNode
            tempNode.prev = newNode
            newNode.rank == newRank
            self.updateRanks()
        #if they want to assign it to any rank inbetween
        else:
            newRank = int(newRankStr)
            #parse through nodes
            currentNode = self.head
            currentNode = currentNode.next
            while(currentNode):
                #if rank of currentNode is equal to newRank then set all the .nexts and .prevs correctly and update ranks and break loop
                if (currentNode.rank == newRank):
                    prevNode = currentNode.prev
                    prevNode.next = newNode
                    currentNode.prev = newNode
                    newNode.prev = prevNode
                    newNode.next = currentNode
                    self.updateRanks()
                    break
                #otherwise keep parsing list
                else:
                    currentNode = currentNode.next

    def addAtEnd(self, company:str, loc:str, title:str, length:str, start:str, pay:str, reqSkills:str, link:str) -> None:
        newNode = Node(company, loc, title, length, start, pay, reqSkills, link)
        currentNode = self.tail
        self.tail = newNode
        currentNode.next = newNode
        newNode.prev = currentNode
        self.updateRanks()
