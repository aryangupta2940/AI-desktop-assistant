def is_armstrong(number):
    digits = [int(d) for d in str(number)]
    power = len(digits)
    total = sum([d ** power for d in digits])
    return total == number

# Print Armstrong numbers in a range ( 1 to 1000)
for num in range(1, 1001):
    if is_armstrong(num):
        print(num)
