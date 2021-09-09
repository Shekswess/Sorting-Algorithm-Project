import time
import json
import numpy as np
# import scipy as sp

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# Styling the look of the whole plot
plt.style.use('dark_background')
plt.rcParams["font.size"] = 14


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # Object for tracking changes # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# This is for the animation but I have some problems with matplotlib and 
# I cannot solve the issue
# Defining the object trackedArray for tracking the changes inside the array 
# during the sorting operation
class TrackedArray():
    
    # Initialization of the Object by giving it the value of the array 
    # we are tracking at the moment of sorting
    def __init__(self, arr):
        self.arr = np.copy(arr)
        self.reset()
    
    # Setting all the variables that we need for tracking to be empty
    # when we are initialising a new array
    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []
        
        
    # The actual function of tracking the changes in the array
    def track(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        self.full_copies.append(np.copy(self.arr))
        
    # Function that tell us what and how we accessed 
    def GetActivity(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])
        
    # Returning the value of the array element at certain position
    def __getitem__(self, key):
        self.track(key, "get")
        return self.arr.__getitem__(key)
    
    # Setting the value of certain array element at certain position
    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.track(key, "set")
        
    # Returning the length of an array
    def __len__(self):
        return self.arr.__len__()
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # Sorting Algorithms # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# best = worst = average = O(n^2), works
# Selection Sort algorithm looks for the min element and puts it
# at the beginning of the array. It has two for loops so it's slow
# but it's easy to implement
def SelectionSort(arr):

    # Accessing the elements of the array, one by one
    for i in range(len(arr)):

        # Creating a variable that is the index of the first min element (it changes by iterations)
        min_ind = i

        # Looking for the index of the min element in the array
        for j in range(i + 1, len(arr)):
            if arr[min_ind] > arr[j]:
                min_ind = j

        # Changing the position of the min element in the array with the element at i position
        arr[i], arr[min_ind] = arr[min_ind], arr[i]

    # Getting back the array
    return arr

# best = O(n), worst = average = O(n^2), works
# Bubble Sort algorithm compares the two adjacent elements and if the first element
# is bigger than the second element, it changes their positions
def BubbleSort(arr):

    # Getting the length of the array
    n = len(arr)

    # Accessing the elements of the array, one by one
    for i in range(n):
        # Accessing again the elents of the array, but the end is new for easier comparing
        for j in range(0, n - i - 1):
            # Comparing the two adjacent elements
            if arr[j] > arr[j + 1]:
                # Changing the position if the first element is bigger than the second
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

    # Returning the array
    return arr

# best = O(n), worst = average = O(n^2), works
# Insertion Sort works similar like Selection Sort, it looks for
# an element with a specific value and searches into the array
# and puts the element on the correct place inside the array
def InsertionSort(arr):
    # Accessing the array, element by element
    for i in range(len(arr)):
        # Setting the first key to be the i element of the array
        key = arr[i]
        # Creating variable j that has the value of i-1
        j = i - 1
        # Loop that loops try the array in reverse and compares
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1

        # Setting the new key
        arr[j + 1] = key

    #Returns the array
    return arr

# best = worst = average = O(n * log(n)), problem with graphing(plotting)
# Merge Sort is an algorithm that divides the array into two halves(right and left) and divides the halves
# again and again. It compares the elements from the halves and puts the smaller ones into a new array that is returned
def MergeSort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        # Recursive call on each half
        MergeSort(left)
        MergeSort(right)

        # Two iterators for traversing the two halves
        i = 0
        j = 0
        
        # Iterator for the main list
        k = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
              # The value from the left half has been used
              arr[k] = left[i]
              # Move the iterator forward
              i += 1
            else:
                arr[k] = right[j]
                j += 1
            # Move to the next slot
            k += 1

        # For all the remaining values
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k]=right[j]
            j += 1
            k += 1

# best = worst = average = O(n * log(n)), works
# Heap Sort algorithm is like building a binary tree, at the end the largest element
# will be the root of the tree
def HeapSort(arr):

    # Function that creates the tree's
    def heapify(arr, n, i):
        # Creating variable largest that has the value of i
        largest = i

        # Creating left(l) and right(r) variables
        l = 2 * i + 1
        r = 2 * i + 2

        # If the length of left/right is smaller than the length of the array and
        # if the element with index larger is smaller than the element with index l/r
        # put larger to be the new l/r
        if l < n and arr[largest] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        # When largest is different than the value of i, then we need to change the elements
        # Plus we call again the function heapify so we can create the new sub tree with largest as root
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    # Variable n that has the value of the length of the array
    n = len(arr)

    # Creating the tree that we need to traverse
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extracting the elements from the tree to an array
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    # Returns the array
    return arr

# best = average = O(n*log(n)), worst = O(n^2), works
# Quick Sort algorithm is similar to Merge Sort, because it divides the array into two halves,
# around a picked element of the array
def QuickSort(start, end, arr):

    # Function that creates the partition of the array and sorts the elements
    def partion(start, end, arr):

        # Picking the element around which we divide the array
        t_ind = start
        t_arr = arr[t_ind]

        # Looping the array
        while start < end:
            # Looping till we are not to the end of the array and till we don't get to the
            # picked element that divides the array
            while start < len(arr) and arr[start] <= t_arr:
                start += 1
            # Same but in reverse order
            while arr[end] > t_arr:
                end -= 1
            # Changing the elements
            if start < end:
                arr[start], arr[end] = arr[end], arr[start]
        # Changing the order of the elements(picked one and the new element)
        arr[end], arr[t_ind] = arr[t_ind], arr[end]
        return end

    # to see if the array doesn't have only one element
    if start < end:
        # Creating the partition
        p = partion(start, end, arr)
        # Calling the sorting for the left part of the partion
        QuickSort(start, p-1, arr)
        # Calling the sorting for the right part of the partion
        QuickSort(p+1, end, arr)

    # Returning the array
    return arr

# best = O(n), worst = average = O(n^2), works
# Gnome Sort is algorithm that is simple(stupid) to implement. It looks two elements
# If the two elements are in the right order it goes forward if not change their positions
def GnomeSort(arr):
    # Preparation for looping the array
    i = 0
    n = len(arr)
    # Looping the array
    while i < n:
        # If it's the first element just skip it and go to the next
        if i == 0:
            i += 1
        # If the left element is equal or larger than the right element, go to the right element
        if arr[i] >= arr[i - 1]:
            i += 1
        # Than do the change and get back to the left element
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1
    # Returns the array
    return arr

# best = average = O(n * log(n)), worst = O(n^2), works
# Shell Sort algorithm is similar like Insertion Sort but much better.
# In Insertion Sort we move element only one position, here we do it much more
def ShellSort(arr):

    # Creating the gap
    n = len(arr) // 2

    # While we have a gap
    while n > 0:
        # Preparing for checking the array from left to right
        i = 0
        j = n
        while j < len(arr):
            # If the left element is larger than the right, change their position
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j += 1
            # Looking from ith index to the left
            # and we change elements that aren't in right order
            k = i
            while k - n > -1:
                if arr[k - n] > arr[k]:
                    arr[k - n], arr[k] = arr[k], arr[k - n]
                k -= 1

        # making the gap smaller
        n //= 2
    # Returns the array
    return arr

# best = O(n), worst = average = O(n^2), works
# CocktailSort algorithm is basicly a variation of Bubble Sort. It is known
# as O(n^2) variation of Bubble Sort, it tranverses the array in both ways alternatively
def CocktailSort(arr):
    # Getting the length of the array
    n=len(arr)
    # Defining a variable true so we can know when to swap elements
    swap = True
    # Setting the start and the end of the array
    start = 0
    end = n-1
    # If the variable is True we enter into the phase of changing positions
    while swap == True:
        # Setting swap to false 
        swap = False
        # We looking for the element if the left element is larget than the right one
        # than change the element
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swap = True
        # If the swap is false than break from the loop        
        if swap == False:
            break
        # Moving to the left from the end of the array
        swap = False
        end = end - 1
        # Same sorting the elements but from right to left
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swap = True

        start = start + 1
    # Returning the array
    return arr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # Helpful Functions for visual implementation # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Function that generates the array and shuffles is
def creatingArray():
    # How many elements   will the array have
    n = 100
    # Creating the array with random elements from 0 to 1000
    arr = np.round(np.linspace(0, 1000, n), 0)
    # Seed so we can always generate the same "random"(pseudo-random) array
    np.random.seed(0)
    # Shuffles the array
    np.random.shuffle(arr)
    # Creating the object TrackedArray of the array we are tracking
    arr = TrackedArray(arr)
    # Return the array
    return arr
   
# Function that creates a graph of the unsorted array (just for testing)
def unsortedArrayGraph(arr):
    n = len(arr)
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, len(arr), 1), arr, align = "edge", width = 0.8, color = 'white')
    ax.set_xlim([0, n])
    ax.set(xlabel = "Index", ylabel = "Value", title = "Unsorted Array")
    plt.savefig('Unsorted_Array.png')

# Function for sorting the array according to the type of algorithm
def sortArray(arr, sorter):
    if sorter.lower() == 'selection':
        # Starting a timer
        t0 = time.perf_counter()
        # Calling the sort algorithm
        SelectionSort(arr)
        # Looking the time at the ending of the sorting algorithm and removing
        # the time from starting the algorithm so I can get the correct time 
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'bubble':
        t0 = time.perf_counter()
        BubbleSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'insertion':
        t0 = time.perf_counter()
        InsertionSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'merge':
        t0 = time.perf_counter()
        MergeSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'heap':
        t0 = time.perf_counter()
        HeapSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'quick':
        t0 = time.perf_counter()
        QuickSort(0, len(arr) - 1, arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'gnome':
        t0 = time.perf_counter()
        GnomeSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'shell':
        t0 = time.perf_counter()
        ShellSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt
    elif sorter.lower() == 'cocktail':
        t0 = time.perf_counter()
        HeapSort(arr)
        dt = time.perf_counter() - t0
        sorterC = sorter.capitalize()
        print(f'------{sorterC} sort------')
        print(f'Array sorted in {dt*1E3:.3f} ms')
        return dt

def sortedArrayGraph(arr, sorter, time):
    sorterC = sorter.capitalize()
    n = len(arr)
    access = len(arr.access_type)
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, len(arr), 1), arr, align = "edge", width = 0.8, color = 'white')
    ax.set_xlim([0, n])
    ax.set(xlabel = "Index", ylabel = "Value", title = f'{sorterC} sort, Access: {access}, Time:{time*1E3:.3f}')
    plt.savefig(f'{sorterC}sort_Array.png')
     
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # Implementation of the Program # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Function that gets average time of all sorting algorithms
def timeAllSort():
    # sorter list so I can tranverse all the algorithms
    sorter = ['selection', 'bubble', 'insertion', 'merge', 'heap', 'quick', 'gnome', 'shell', 'cocktail']
    # List I can use for creating dictionary for the data
    sorterJ = ['Selection Sort', 'Bubble Sort', 'Insertion Sort', 'Merge Sort', 'Heap Sort', 'Quick Sort', 'Gnome Sort', 'Shell Sort', 'Cocktail Sort']
    # Lists for saving the average times of sorting algorithms and all the times for 10 loops
    times = []
    timesD = []
    # Going throu all the algorithms
    for s in sorter:
        i = 0
        dt = []
        while i < 10:
            # Need to reset the array every time I run a sorting algorithm
            arr = creatingArray()
            t = sortArray(arr, s)
            dt.append(t)
            i += 1
        average = float(sum(dt) / len(dt))
        times.append(average)
        timesD.append(dt)
    # Creating the dictionaries
    info = {}
    infoD = {}
    i = 0
    # Updating the dictionaries
    while i < 9:
        info.update({sorterJ[i] : times[i]})
        infoD.update({sorterJ[i] : timesD[i]})
        i += 1
    # Putting all the data into json files
    with open('AverageTimeSorting.json', "w") as ats:
        json.dump(info, ats, indent=4, separators=(',', ': '))
    with open('TimeSortingDetailed.json', "w") as  atsd:
        json.dump(infoD, atsd, indent=4, separators=(',', ': '))
    
# Function that draws the arrays before and after the sorting
def drawSort():
    # Getting the sorter algorithm that I will use
    sorter = input('What sort algorithm you want to try ? ')
    arr = creatingArray()
    unsortedArrayGraph(arr)
    time = sortArray(arr, sorter)
    sortedArrayGraph(arr, sorter, time)


print('Welcome to the Sorting Algorithm System !!!')
print('')
print('# # # # # # # # # # # # # # # # # # # #')
print('')
while True:
    print('For the execution time for all the algorithms write ALGORITHMS !')
    print('For the details for one algorithm write ALGORITHM !')
    print('For exiting the system write EXIT !')
    inp = input()
    if inp.upper() == 'ALGORITHMS':
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        timeAllSort()
        print('')
        print('You have all the information in the JSON files !')
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        print('')
    elif inp.upper() == 'ALGORITHM':
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        drawSort()
        print('')
        print('You have all the information on the plot picture !')
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        print('')
    else:
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        print('')
        print('You wrote a wrong command or you wrote EXIT !!!')
        print('Rerun the system again !')
        break
        print('')
        print('# # # # # # # # # # # # # # # # # # # #')
        print('')
    