import pandas as pd
import linkedList as ll
from Python.pythonToGoogleSheets.internshipInfo.linkedList import DoublyLinkedList


def putDataInLL():
    #read the file
    data = pd.read_csv("file.csv")
    #read the first line and assign correct values to correct variables
    newComp = data["Company"].iloc[0]
    newLoc = data["Location"].iloc[0]
    newTitle = data["Title"].iloc[0]
    newLen = data["Length"].iloc[0]
    newStart = data["Start Date"].iloc[0]
    newPay = data["Pay"].iloc[0]
    newReq = data["Required Skills"].iloc[0]
    newLink = data["Link"].iloc[0]
    #make the first node
    firstNode = ll.Node(newComp, newLoc, newTitle, newLen, newStart, newPay, newReq, newLink)
    #make the linked list
    linkedList = ll.DoublyLinkedList(firstNode)
    #for every other line in the data sheet copy the same steps as above and add to linked list
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
    return linkedList

def putLLInData(linkedList:DoublyLinkedList) -> None:
    #make a data list
    dataList = []
    currentNode = linkedList.head
    #enter all the data in the nodes to the data list
    while(currentNode):
        dataList.append({
            'Company': currentNode.company,
            'Location': currentNode.loc,
            'Title': currentNode.title,
            'Length': currentNode.length,
            'Start Date': currentNode.start,
            'Pay': currentNode.pay,
            'Required Skills': currentNode.requiredSkills,
            'Link': currentNode.link
        })
        currentNode = currentNode.next
    #create pandas data frame
    data = pd.DataFrame(dataList)
    #save the new data to the csv file
    data.to_csv("file.csv", index=False)