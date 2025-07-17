'''Yes I do realize that this is really over engineering to just add internship oppotunities and rank them in a
 Google Sheet, but this is good practice with data structures like linked lists, manipulating data, and
 using the Google API. This also helps practice principle of least privilege'''

import googleTools as gt
import linkedList as ll

gt.updateFile()
firstNode = ll.Node("Optiver", "Chicago", "Soft. Eng.", "?", "?", "?", "C++, Python, Java, CS fundamentals", "link here")
linkedList = ll.DoublyLinkedList(firstNode)
linkedList.printLL()
linkedList.addNewInternship()
linkedList.printLL()

