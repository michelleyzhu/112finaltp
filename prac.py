
# this function tells us if n is even AND positive AND an integer
# this function should return true or false
def isEvenPosInt(n):
    if(isinstance(n,int) and n > 0 and n%2 == 0):
        return True
    return False


print(isEvenPosInt(8))