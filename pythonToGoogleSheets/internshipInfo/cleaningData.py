import pandas as pd
import linkedList as ll

def putDataInLL() -> None:
    data = pd.read_csv("file.csv")
    newComp = data["Company"].iloc[0]
    newLoc = data["Location"].iloc[0]
    newTitle = data["Title"].iloc[0]
    newLen = data["Length"].iloc[0]
    newStart = data["Start Date"].iloc[0]
    newPay = data["Pay"].iloc[0]
    newReq = data["Required Skills"].iloc[0]
    newLink = data["Link"].iloc[0]
    firstNode = ll.Node(newComp, newLoc, newTitle, newLen, newStart, newPay, newReq, newLink)
    linkedList = ll.DoublyLinkedList(firstNode)
    for i in range(len(data)):
        if i != 0:
            newComp = data["Company"].iloc[i]
            newLoc = data["Location"].iloc[i]
            newTitle = data["Title"].iloc[i]
            newLen = data["Length"].iloc[i]
            newStart = data["Start Date"].iloc[i]
            newPay = data["Pay"].iloc[i]
            newReq = data["Required Skills"].iloc[i]
            newLink = data["Link"].iloc[i]
            linkedList.addAtEnd(newComp, newLoc, newTitle, newLen, newStart, newPay, newReq, newLink)