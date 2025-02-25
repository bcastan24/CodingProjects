#A program to generate passwords
#It will generate a password that is at least 8 characters long, has at least: one upper case letter, one lower case letter, one number, and one special character
import random
#need to make lists of all upper case letters, all lower case letters, all numbers, and all special characters
upperCaseList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
lowerCaseList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']
numbersList = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
specialCharList = ['!', '&', '*']
#I will add all the randomized characters to a list, so I can randomize the order of the characters
#so there is no predictable order for password cracking
passList = []
passList.append(random.choice(upperCaseList))
passList.append(random.choice(lowerCaseList))
passList.append(random.choice(numbersList))
passList.append(random.choice(specialCharList))
#four of the characters will be one of each type, the other four will be randomized
#0=uppercase 1=lowercase 2=number 3=special char
remainingCharsList = [0, 1, 2, 3]
#I will ask the user how long they want the password to be, has to be between 8 and 15 chars
lenOfPass = input("How long do you want the password to be?(Has to be between 8 and 15 characters): ")
if int(lenOfPass) >= 8 and int(lenOfPass) <= 15:
    remainingChars = int(lenOfPass) - 4
    x = 0
    #I will make a while loop to randomize remaining characters
    while x < remainingChars:
        y = random.choice(remainingCharsList)
        if y == 0:
            passList.append(random.choice(upperCaseList))
            x = x+1
        elif y == 1:
            passList.append(random.choice(lowerCaseList))
            x = x+1
        elif y == 2:
            passList.append(random.choice(numbersList))
            x = x+1
        else:
            passList.append(random.choice(specialCharList))
            x = x+1
    item = random.choice(passList)
    password = str(item)
    passList.remove(item)
    while len(passList) != 0:
        item = random.choice(passList)
        password += str(item)
        passList.remove(item)
    print(password)

else:
    print("You entered a number that is not in the range of 8-15, please restart program and enter a number within that range.")

