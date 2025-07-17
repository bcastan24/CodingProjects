'''Yes I do realize that this is really over engineering to just add internship oppotunities and rank them in a
 Google Sheet, but this is good practice with data structures like linked lists, manipulating data, and
 using the Google API. This also helps practice principle of least privilege and reading documentation, for all the code
 for the Google API I read the documentation to figure what to write and why. I left out some key files that
 someone would need to actually run this program but I just hid the stuff that had sensitive information'''

import googleTools as gt
import cleaningData as ll

file = gt.updateFile()
linkedList = ll.putDataInLL()
shouldAdd = input("Would you like to add an internship? (Yes or No): ")
if shouldAdd.lower() == "yes":
    while (shouldAdd.lower() == "yes"):
        linkedList.addNewInternship()
        shouldAdd = input("Would you like to add another internship? (Yes or No): ")
    ll.putLLInData(linkedList)
    gt.uploadFile()




