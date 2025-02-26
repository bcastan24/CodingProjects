#A program to crack a password
#note that this password cracker just brute forces passwords
#for any password between 8 and 15 characters this will take a VERY long time
import time

password = input("Please enter the password you would like to test: ")
start = time.time()
#All possible characters that could be in password
possibleCharsList = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!&*'

guess = []
for val in range(8, 15):
    a = [i for i in possibleCharsList]
    for y in range(val):
        a = [x+i for i in possibleCharsList for x in a]
    guess = guess+a
    if password in guess:
        break
end = time.time()
clock = str(end-start)

print("Your password: " + password)
print("Time taken: " + clock)
