class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def insertNode(self, x):
        node = self
        while node.next is not None:
            node = node.next
        node.next = ListNode(x)

    def printList(self):
        node = self
        myList = []
        while node is not None:
            myList.append(node.val)
            node = node.next
        print(myList)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x > 2 ** 31 - 1 or x < -2 ** 31:
            return 0
        sign = 1 if x >= 0 else -1
        x = x * sign
        output = 0
        while x / 10 > 0 :
            digit = x % 10
            output = output * 10 + digit
            x = int(x / 10)
        output = output * sign
        if output > 2 ** 31 - 1 or output < -2 ** 31:
            return 0
        return output

    def isInt(self, mychar):
        if ord(mychar) - ord('0') <= 9 and ord(mychar) - ord('0') >= 0:
            return True
        else:
            return False

    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        s = Solution()
        myAtoi_int = 0
        sign = 0
        # myAtoi_flag = False when we have not seen the first digit yet
        myAtoi_flag = False
        for i in range(0, len(str)):
            if str[i] != ' ' and str[i] != '-' and str[i] != '+' and \
                    not(s.isInt(str[i])):
                break
            if sign != 0 and not s.isInt(str[i]):
                break
            if str[i] == '-' and not myAtoi_flag:
                if sign == 0:
                    sign = -1
                else:
                    break
            if str[i] == '+' and not myAtoi_flag:
                if sign == 0:
                    sign = 1
                else:
                    break
            if s.isInt(str[i]):
                myAtoi_int = myAtoi_int * 10 + ord(str[i]) - ord('0')
                myAtoi_flag = True
            if myAtoi_flag and not s.isInt(str[i]):
                break
        sign = -1 if sign == -1 else 1
        if myAtoi_flag:
            out = myAtoi_int * sign
            if out < -2**31:
                return -2**31
            elif out > 2**31 - 1:
                return 2**31 -1
            else:
                return out
        else:
            return 0

    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        reversed = 0
        temp = x + 0
        if x < 0:
            return False
        while(temp > 0):
            digit = temp % 10
            reversed = reversed * 10 + digit
            temp = int(temp / 10)
        return x == reversed

    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        max_water = 0
        (left, right) = (0, len(height) - 1)
        while (left < right):
            max_water = max(max_water, min(height[left], height[right]) * (right - left))
            if (height[left] < height[right]):
                left += 1
            else:
                right -= 1
        return max_water

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        basic_dict = {1: "I", 5: "V", 10: "X", 50: "L", 100: "C", 500: "D", 1000: "M"}
        keys = [1000, 100, 10, 1]
        output_roman = ""
        for i in range(0, len(keys)):
            key = keys[i]
            digit = int(num / key)
            if (digit > 0):
                if (digit == 5):
                    output_roman = output_roman + basic_dict[digit * key]
                elif (digit == 9):
                    output_roman = output_roman + basic_dict[key] + basic_dict[10 * key]
                elif (digit == 4):
                    output_roman = output_roman + basic_dict[key] + basic_dict[5 * key]
                else:
                    if (digit > 5):
                        output_roman = output_roman + basic_dict[5 * key]
                        rep = int((num - 5 * key) / key)
                        for j in range(0, rep):
                            output_roman = output_roman + basic_dict[key]
                    else:
                        for j in range(0, digit):
                            output_roman = output_roman + basic_dict[key]
            num = num % key
        return output_roman

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        common_prefix = ""
        if not strs:
            return common_prefix
        min_strlen = min([len(x) for x in strs])
        for j in range(0, min_strlen):
            for i in range(0, len(strs) - 1):
                if (strs[i][j] != strs[i + 1][j]):
                    return common_prefix
            common_prefix = common_prefix + strs[0][j]
        return common_prefix

    # This solution throws a TLE for long lists, although being O(N*N) complexity
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def hashFn(triple):
            return (2**triple[0]) * (3**triple[1]) * (5**triple[2])

        idx_set = {}
        hash_set = set()
        idx_result = []
        result = []
        nums.sort()
        for i in range(0, len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if -(nums[i] + nums[j]) in idx_set:
                    idx_set[-(nums[i] + nums[j])].append({i, j})
                else:
                    idx_set[-(nums[i] + nums[j])] = [{i, j}]

        for i in range(0, len(nums)):
            if nums[i] in idx_set:
                for pair_idx in idx_set[nums[i]]:
                    if i not in pair_idx:
                        triple_idx = set(pair_idx)
                        triple_idx.add(i)
                        idx_result.append(triple_idx)

        for idx in idx_result:
            triple_idx = list(idx)
            triple = [nums[triple_idx[0]], nums[triple_idx[1]], nums[triple_idx[2]]]
            triple.sort()
            hashVal = hashFn(triple)
            if hashVal not in hash_set:
                hash_set.add(hashVal)
                result.append(triple)

        return result

    # This solution throws a TLE for long lists, although being O(N*N) complexity
    def threeSum2(self, nums):
        def hashFn(triple):
            return str(triple[0]) + "-" + str(triple[1]) + "-" + str(triple[2])
        result = []
        nums.sort()
        sum_set = {}
        for i in range(0, len(nums) - 1):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if -(nums[i] + nums[j]) == nums[j]:
                    if j == len(nums) - 1 or nums[j] != nums[j + 1]:
                        continue
                if -(nums[i] + nums[j]) == nums[i]:
                    if nums[i + 1] != nums[i]:
                        continue
                if -(nums[i] + nums[j]) in sum_set:
                    sum_set[-(nums[i] + nums[j])].append([nums[i], nums[j]])
                else:
                    sum_set[-(nums[i] + nums[j])] = [[nums[i], nums[j]]]
        hash_set = set()
        for i in range(0, len(nums)):
            num = nums[i]
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if num in sum_set:
                for pair in sum_set[num]:
                    triple = list(pair)
                    triple.append(num)
                    triple.sort()
                    hashVal = hashFn(triple)
                    if hashVal not in hash_set:
                        hash_set.add(hashVal)
                        result.append(triple)
        return result

    def threeSum3(self, nums):
        res = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            l, r = i + 1, len(nums) - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1;
                    r -= 1
        return res

    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def distance(a, b):
            dist = a - b if a >= b else b - a
            return dist
        nums.sort()
        closest_sum = 2**32-1
        for i in range(0, len(nums) - 2):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                current_dist = distance(sum, target)
                closest_dist = distance(closest_sum, target)
                if current_dist < closest_dist:
                    closest_sum = sum
                if sum > target:
                    r -= 1
                elif sum < target:
                    l += 1
                else:
                    return sum
        return closest_sum

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        digitsToLetterMap = {"2":["a", "b", "c"], "3":["d", "e", "f"], "4":["g", "h", "i"], "5":["j", "k", "l"], \
                             "6": ["m", "n", "o"], "7":["p", "q", "r", "s"], "8":["t", "u", "v"], "9":["w", "x", "y", "z"]}
        letterCombs = []
        for i in range(0, len(digits)):
            mappedLetters = digitsToLetterMap[digits[i]]
            if letterCombs == []:
                letterCombs = list(mappedLetters)
            else:
                newLetterCombs = []
                for comb in letterCombs:
                    for letter in mappedLetters:
                        newLetterCombs.append(comb + letter)
                letterCombs = list(newLetterCombs)
        return letterCombs

    # This solution throws a TLE. It uses the threeSum

    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def threeSumTarget(nums, target):
            res = []
            for i in range(len(nums) - 2):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                l, r = i + 1, len(nums) - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s < target:
                        l += 1
                    elif s > target:
                        r -= 1
                    else:
                        res.append((nums[i], nums[l], nums[r]))
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1
            return res

        def myHashFn(quadruple):
            return str(quadruple[0]) + "-" + str(quadruple[1]) + "-" + str(quadruple[2]) + "-" + str(quadruple[3])

        def mySort(triple, num):
            if num < triple[0]:
                return num + triple
            elif num < triple[1]:
                return [triple[0], num, triple[1], triple[2]]
            elif num < triple[2]:
                return [triple[0], triple[1], num, triple[2]]
            else:
                return triple + num

        nums.sort()
        result = []
        hash_set = set()
        for i in range(0, len(nums)):
            rest = list(nums)
            rest.remove(nums[i])
            threeSumTriples = threeSumTarget(rest, target - nums[i])
            if threeSumTriples != []:
                for triple in threeSumTriples:
                    quadruple = mySort(triple, nums[i])
                    hashval = myHashFn(quadruple)
                    if hashval not in hash_set:
                        hash_set.add(hashval)
                        result.append(quadruple)
        return result

    def fourSum2(self, nums, target):
        def myHashFn(quadruple):
            return str(quadruple[0]) + "-" + str(quadruple[1]) + "-" + str(quadruple[2]) + "-" + str(quadruple[3])
        nums.sort()
        pair_idx_set = {}
        result_idx = []
        my_set = set()
        result = []
        for i in range(0, len(nums) - 1):
            # if i > 0 and nums[i] == nums[i - 1]:
            #     continue
            for j in range(i + 1, len(nums)):
                # if j > i + 1 and nums[j] == nums[j - 1]:
                #     continue
                if nums[i] + nums[j] not in pair_idx_set:
                    pair_idx_set[nums[i] + nums[j]] = [[i, j]]
                else:
                    pair_idx_set[nums[i] + nums[j]].append([i, j])

        for pair_sum in pair_idx_set:
            idx_set1 = pair_idx_set[pair_sum]
            res = target - pair_sum
            if res in pair_idx_set and res >= pair_sum:
                idx_set2 = pair_idx_set[res]
            else:
                continue
            for idx1 in idx_set1:
                for idx2 in idx_set2:
                    if idx1[0] != idx2[0] and idx1[0] != idx2[1] and \
                            idx1[1] != idx2[0] and idx1[1] != idx2[1]:
                        result_idx.append([idx1[0], idx1[1], idx2[0], idx2[1]])

        for idx_set in result_idx:
            quadruple_idx = list(idx_set)
            quadruple_idx.sort()
            quadruple = [nums[quadruple_idx[0]], nums[quadruple_idx[1]], nums[quadruple_idx[2]], nums[quadruple_idx[3]]]
            hashVal = myHashFn(quadruple)
            if hashVal not in my_set:
                result.append(quadruple)
                my_set.add(hashVal)

        return result

    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        arrOfListNodes = []
        pt = head
        while pt is not None:
            arrOfListNodes.append(pt)
            pt = pt.next
        if n == 1:
            idx = len(arrOfListNodes) - n
            if idx == 0:
                head = None
            else:
                prevNode = arrOfListNodes[idx - 1]
                prevNode.next = None
        elif n == len(arrOfListNodes):
            head = head.next
        else:
            idx = len(arrOfListNodes) - n
            prevNode = arrOfListNodes[idx - 1]
            nextNode = arrOfListNodes[idx + 1]
            prevNode.next = nextNode
        return head

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # if s == "":
        #     return True
        closing_paren = {"]": "[", ")": "(", "}": "{"}
        paren_stack = []
        for i in range(0, len(s)):
            if s[i] in closing_paren:
                if paren_stack == []:
                    return False
                if paren_stack[-1] == closing_paren[s[i]]:
                    paren_stack.pop()
                else:
                    return False
            else:
                paren_stack.append(s[i])

        if paren_stack == []:
            return True
        else:
            return False

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def recurParen(o, c, parenStr, result):
            if o == 0 and c == 0:
                result.append(parenStr)
                return
            if o == 0:
                recurParen(o, c - 1, parenStr + ")", result)
            elif o == c:
                parenStr += "("
                recurParen(o - 1, c, parenStr, result)
            else:
                recurParen(o - 1, c, parenStr + "(", result)
                recurParen(o, c - 1, parenStr + ")", result)
        result = []
        recurParen(n, n, "", result)
        return result

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == []:
            return 0
        (pt1, pt2) = (1, 1)
        count_distinct = 1
        while pt2 < len(nums) and pt1 < len(nums):
            if nums[pt1] > nums[pt1 - 1]:
                pt1 += 1
                count_distinct += 1
            elif nums[pt1] <= nums[pt1 - 1] < nums[pt2]:
                nums[pt1], nums[pt2] = nums[pt2], nums[pt1]
                pt1 += 1
                pt2 += 1
                count_distinct += 1
            else:
                pt2 += 1
        return count_distinct

    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return head
        if head.next == None:
            return head
        pt = head
        head = pt.next
        pt.next = head.next
        head.next = pt
        prevNode = pt
        while pt.next:
            pt = pt.next
            if pt.next:
                nextNode = pt.next.next
                prevNode.next = pt.next
                pt.next = nextNode
                prevNode.next.next = pt
                prevNode = prevNode.next.next
        return head

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m - 1] <= nums2[n - 1]:
                nums1[n + m - 1] = nums2[n - 1]
                n -= 1
            else:
                nums1[n + m - 1] = nums1[m - 1]
                m -= 1
        if n > 0:
            nums1[0:n] = nums2[0:n]

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        import heapq
        minheap = []
        result = ListNode(0)
        pt = result
        for i in range(0, len(lists)):
            if lists[i]:
                heapq.heappush(minheap, (lists[i].val, i))
                lists[i] = lists[i].next
        while minheap:
            minValNode = heapq.heappop(minheap)
            idx = minValNode[1]
            pt.next = ListNode(minValNode[0])
            pt = pt.next
            if lists[idx]:
                heapq.heappush(minheap, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return result.next

    def reverseLinkedList(self, head):
        def reverseNodes(prevNode, curNode, nextNode):
            if nextNode == None:
                curNode.next = prevNode
                return curNode
            curNode.next = prevNode
            return reverseNodes(curNode, nextNode, nextNode.next)
        return reverseNodes(None, head, head.next)

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        def reverseList(start, end):
            def reverseNodes(prevNode, curNode, nextNode, end):
                if nextNode == end:
                    curNode.next = prevNode
                    return curNode
                curNode.next = prevNode
                return reverseNodes(curNode, nextNode, nextNode.next, end)
            head = start.next
            return reverseNodes(end, head, head.next, end)
        counter = 0
        end = head
        dummyStart = ListNode(0)
        dummyStart.next = head
        start = dummyStart
        while end != None:
            counter += 1
            end = end.next
            if counter == k:
                nextStart = start.next
                start.next = reverseList(start, end)
                start = nextStart
                counter = 0
        return dummyStart.next

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if nums == []:
            return 0
        l, r = 0, len(nums) - 1
        while l < r:
            if nums[l] == val:
                if nums[r] != val:
                    nums[l], nums[r] = nums[r], nums[l]
                    r -= 1
                    l += 1
                else:
                    r -= 1
            else:
                l += 1
        if nums[l] == val:
            return l
        else:
            return l + 1

    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if divisor == 0:
            return 0
        sign = 1 if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0) else -1
        dividend = -dividend if dividend < 0 else dividend
        divisor = -divisor if divisor < 0 else divisor
        def divideRec(dividend, divisor):
            if (dividend < divisor):
                return 0
            if (dividend == divisor):
                return 1
            resid = dividend
            mydivisor = divisor
            res = 0
            quotient = 0
            while resid >= mydivisor:
                resid -= mydivisor
                mydivisor += mydivisor
                if quotient == 0:
                    quotient += 1
                else:
                    quotient += quotient
                res += quotient
            return res + divideRec(resid, divisor)

        res = divideRec(dividend, divisor) * sign

        if res < -2 ** 31 or res > 2 ** 31 - 1:
            return 2 ** 31 - 1
        else:
            return res

    def fib(self, n):
        def fibmemo(n, memo):
            if n <= 0:
                return 0
            elif n == 1:
                memo[n - 1]  = 1
                return 1
            elif memo[n - 1] == 0:
                memo[n - 1] = fibmemo(n-1, memo) + fibmemo(n-2, memo)
            return memo[n - 1]
        memo = [0] * n
        return fibmemo(n, memo)

    def findSubset(self, target, nums):
        def findSubsetRec(target, nums, result, myset):
            if target == 0 and len(nums) >= 0:
                result.append(myset)
                return
            elif len(nums) == 1 and nums[0] == target:
                myset.append(nums[0])
                result.append(myset)
                return
            elif len(nums) ==1 and nums[0] != target:
                return
            else:
                for i in range(0, len(nums)):
                    curr_set = list(myset)
                    curr_set.append(nums[i])
                    findSubsetRec(target - nums[i], nums[i+1:], result, curr_set)
        result = []
        findSubsetRec(target, nums, result, [])
        return result

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        def reverseList(nums, start, end):
            while (start < end):
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

        right_index = len(nums) - 1
        while right_index > 0 and nums[right_index] <= nums[right_index - 1]:
            right_index -= 1
        if right_index == 0:
            reverseList(nums, 0, len(nums) - 1)
        else:
            swap_ind1 = right_index - 1
            while right_index < len(nums) and (nums[right_index] > nums[swap_ind1]):
                right_index += 1
            swap_ind2 = right_index - 1
            nums[swap_ind1], nums[swap_ind2] = nums[swap_ind2], nums[swap_ind1]
            reverseList(nums, swap_ind1 + 1, len(nums) - 1)

    # Input: "(()"
    # Output: 2
    # Explanation: The longest valid parentheses substring is "()"
    # Input: ")()())"
    # Output: 4
    # Explanation: The longest valid parentheses substring is "()()"
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = list()
        maxLen = 0
        subStrLen = dict.fromkeys(range(0, len(s)), 0)
        i = 0
        isFirstOpenParen = False
        firstOpenParen = -1
        while i < len(s):
            if s[i] == ")" and len(stack) > 0:
                stack.pop()
                if len(stack) == 0:
                    lastStart = firstOpenParen
                else:
                    lastStart = stack[-1] + 1
                subStrLen[lastStart] = i - lastStart + 1
            elif s[i] == "(":
                if not isFirstOpenParen:
                    isFirstOpenParen = True
                    firstOpenParen = i
                stack.append(i)
            elif s[i] == ")" and len(stack) == 0:
                isFirstOpenParen = False
            i += 1
        for key in subStrLen.keys():
            maxLen = max(maxLen, subStrLen[key])
        return maxLen

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        start = 0
        end = len(nums) - 1
        if start == end:
            return 0 if nums[start] == target else -1
        while start < end:
            mid = int((start + end) / 2)
            if nums[start] == target: return start
            if nums[mid] == target: return mid
            if nums[end] == target: return end
            if mid == start: return -1
            if nums[start] < nums[end]:
                if target >= nums[mid]:
                    start = mid
                else:
                    end = mid
            else:
                if nums[mid] <= nums[end]:
                    if target >= nums[mid] and target <= nums[end]:
                        start = mid
                    else:
                        end = mid
                else:
                    if target <= nums[end] or target >= nums[mid]:
                        start = mid
                    else:
                        end = mid
        return -1

    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def findMaxHeight(root):
            if root is None:
                return 0
            if root.left is not None and root.right is not None:
                return max(findMaxHeight(root.left), findMaxHeight(root.right)) + 1
            elif root.left is not None:
                return findMaxHeight(root.left) + 1
            elif root.right is not None:
                return findMaxHeight(root.right) + 1
            else:
                return 1

        maxHeight = findMaxHeight(root)
        res = list()
        def findLevelNode(root, level, levelSet):
            if root is None: return
            if level == 1:
                levelSet.append(root.val)
                return
            else:
                findLevelNode(root.left, level - 1, levelSet)
                findLevelNode(root.right, level - 1, levelSet)

        for lvl in range(maxHeight, 0, -1):
            levelSet = list()
            findLevelNode(root, lvl, levelSet)
            res.append(levelSet)
        return res

    def levelOrderBottom2(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        queue =  []
        lOrder = []

        if root is not None:
            queue.append(root)

        while len(queue) > 0:
            llist = []
            size = len(queue)
            for i in range(size):
                curNode = queue.pop()
                if curNode.left is not None :
                    queue.insert(0,curNode.left)
                if curNode.right is not None :
                    queue.insert(0,curNode.right)
                llist.append(curNode.val)
            lOrder.insert(0, llist)
        return lOrder

    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def findMinDepth(root):
            if root is None: return 0
            if root.left is not None and root.right is not None:
                return min(findMinDepth(root.left), findMinDepth(root.right)) + 1
            elif root.left is not None:
                return findMinDepth(root.left) + 1
            elif root.right is not None:
                return findMinDepth(root.right) + 1
            else:
                return 1

        return findMinDepth(root)

    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        queue = []
        res = []
        if root is not None:
            queue.append((0, root))
        lvlSeenSoFar = -1
        while len(queue) > 0:
            (curLvl, curNode) = queue.pop()
            if curLvl > lvlSeenSoFar:
                res.append(curNode.val)
                lvlSeenSoFar = curLvl
            if curNode.right is not None:
                queue.insert(0, (curLvl + 1, curNode.right))
            if curNode.left is not None:
                queue.insert(0, (curLvl + 1, curNode.left))
        return res

    def rightSideView2(self, root):
        if root == None:
            return []
        q = [root]
        s = []
        while len(q) > 0:
            length = len(q)
            for i in range(length):
                current = q.pop(0)
                if current.left:
                    q.append(current.left)
                if current.right:
                    q.append(current.right)

            s.append(current.val)
        return s

    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        lvl = 0
        queue = [root]
        res = []
        while len(queue) > 0:
            curLvl = []
            for i in range(0, len(queue)):
                if lvl % 2 == 1:
                    curNode = queue.pop()
                    curLvl.append(curNode.val)
                    if curNode.right:
                        queue.insert(0, curNode.right)
                    if curNode.left:
                        queue.insert(0, curNode.left)
                else:
                    curNode = queue.pop(0)
                    curLvl.append(curNode.val)
                    if curNode.left:
                        queue.append(curNode.left)
                    if curNode.right:
                        queue.append(curNode.right)

            res.append(curLvl)
            lvl += 1
        return res

    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        def findNeighbors(row, col, maxRow, maxCol):
            neighbors = []
            for (r, c) in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
                if 0 <= r < maxRow and 0 <= c < maxCol:
                    neighbors.append((r, c))
            return neighbors

        queue = []
        INT_MAX = 10000
        maxRow = len(matrix)
        maxCol = len(matrix[0])
        res = [[INT_MAX for col in range(0, maxCol)] for row in range(0, maxRow)]
        seenBefore = set()
        for row in range(0, len(matrix)):
            for col in range(0, len(matrix[row])):
                if matrix[row][col] == 0:
                    queue.append((row, col))
                    res[row][col] = 0
                    seenBefore.add((row, col))
        while len(queue) > 0:
            for i in range(0, len(queue)):
                (row, col) = queue.pop(0)
                neighbors = findNeighbors(row, col, maxRow, maxCol)
                minDist = INT_MAX
                for (r, c) in neighbors:
                    if res[r][c] == INT_MAX and (r, c) not in seenBefore:
                        queue.append((r, c))
                        seenBefore.add((r, c))
                    minDist = min(minDist, res[r][c])
                if matrix[row][col] == 0:
                    res[row][col] = 0
                else:
                    res[row][col] = minDist + 1
        return res

    def mergeSort(self, nums):
        def sortedMerge(nums1, nums2):
            sortedNums = []
            i, j = 0, 0
            while i < len(nums1) and j < len(nums2):
                if nums1[i] < nums2[j]:
                    sortedNums.append(nums1[i])
                    i += 1
                else:
                    sortedNums.append(nums2[j])
                    j += 1
            if i < len(nums1):
                sortedNums.extend(nums1[i:])
            if j < len(nums2):
                sortedNums.extend(nums2[j:])
            return sortedNums

        def mergeSplit(nums, start, end):
            if start == end:
                return [nums[start]]
            if end - start == 1:
                if nums[start] <= nums[end]:
                    return [nums[start], nums[end]]
                else:
                    return [nums[end], nums[start]]
            mid = int((start + end) / 2)
            sortedNums1 = mergeSplit(nums, start, mid)
            sortedNums2 = mergeSplit(nums, mid + 1, end)
            return sortedMerge(sortedNums1, sortedNums2)

        return mergeSplit(nums, 0, len(nums) - 1)

    def flipTree(self, root):
        def flip(root):
            if not root:
                return root
            if root.left and root.right:
                newleft = flip(root.right)
                newright = flip(root.left)
                root.left = newleft
                root.right = newright
                return root
            elif root.left:
                newright = flip(root.left)
                root.right = newright
                root.left = None
                return root
            elif root.right:
                newleft = flip(root.right)
                root.left = newleft
                root.right = None
                return root
            else:
                return root
        return flip(root)

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isMirror(left, right):
            if left is None and right is None:
                return True
            if left is None and right:
                return False
            if left and right is None:
                return False
            if left.val == right.val:
                return isMirror(left.left, right.right) and \
                       isMirror(left.right, right.left)
            else:
                return False
        if root is None: return True
        return isMirror(root.left, root.right)

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def getMiddleNode(head):
            fast = head
            slow = head
            if head is None:
                return None
            while slow.next and fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def sortedMerge(head1, head2):
            dummyStart = ListNode(0)
            cur = dummyStart
            while head1 and head2:
                if head1.val <= head2.val:
                    cur.next = head1
                    head1 = head1.next
                else:
                    cur.next = head2
                    head2 = head2.next
                cur = cur.next
            if head1:
                cur.next = head1
            else:
                cur.next = head2
            return dummyStart.next

        def mergeSplit(head):
            if head is None or head.next is None:
                return head
            mid = getMiddleNode(head)
            head1 = head
            head2 = mid.next
            mid.next = None
            newhead1 = mergeSplit(head1)
            newhead2 = mergeSplit(head2)
            return sortedMerge(newhead1, newhead2)

        return mergeSplit(head)

    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def isBalancedRec(root):
            if root.left is None and root.right is None:
                return 1
            leftHeight, rightHeight = 1, 1
            if root.left:
                tmpLeftHeight = isBalancedRec(root.left)
                if tmpLeftHeight == -1:
                    leftHeight = -1
                else:
                    leftHeight = tmpLeftHeight + 1
            if root.right:
                tmpRightHeight = isBalancedRec(root.right)
                if tmpRightHeight == -1:
                    rightHeight = -1
                else:
                    rightHeight = tmpRightHeight + 1
            if leftHeight == -1 or rightHeight == -1:
                return -1
            elif abs(leftHeight - rightHeight) <= 1:
                return max(leftHeight, rightHeight)
            else:
                return -1
        if root is None:
            return True
        return isBalancedRec(root) != -1

    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        else:
            res = [[1]]
        for i in range(1, numRows):
            prevRow = res[i-1]
            curRow = [0] * (i+1)
            for j in range(0, i+1):
                if j == 0 or j == i:
                    curRow[j] = 1
                else:
                    curRow[j] = prevRow[j - 1] + prevRow[j]
            res.append(curRow)
        return res

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        def DFS(courseToCheck, preReqMap, checked):
            if checked[courseToCheck] == 1:
                return True
            # There is a cycle. We are already checking courseToCheck
            if checked[courseToCheck] == -1:
                return False

            checked[courseToCheck] = -1
            if courseToCheck in preReqMap:
                for prereq in preReqMap[courseToCheck]:
                    if not DFS(prereq, preReqMap, checked):
                        return False
            checked[courseToCheck] = 1
            return True


        preReqMap = dict()
        for (course, prereq) in prerequisites:
            if course in preReqMap:
                preReqMap[course].append(prereq)
            else:
                preReqMap[course] = [prereq]

        checkedCourses = [0] * numCourses
        for course in preReqMap:
            if not DFS(course, preReqMap, checkedCourses):
                return False
        return True

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        def buildSubTree(preorder, inorder):
            if len(preorder) == 0:
                return None
            if len(inorder) == 0:
                return None
            if len(preorder) == 1 and len(inorder) == 1:
                return TreeNode(preorder[0])

            rootNode = preorder[0]
            root_index = 0
            while inorder[root_index] != rootNode:
                root_index += 1
            leftInOrderNodes = inorder[0:root_index]
            rightInOrderNodes = inorder[root_index+1:]
            leftPreOrderNodes = preorder[1:1+len(leftInOrderNodes)]
            rightPreOrderNodes = preorder[1+len(leftInOrderNodes):]
            root = TreeNode(rootNode)
            root.left = buildSubTree(leftPreOrderNodes, leftInOrderNodes)
            root.right = buildSubTree(rightPreOrderNodes, rightInOrderNodes)
            return root
        return buildSubTree(preorder, inorder)

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def buildTreeFromSortedArray(nums, start, end):
            if start == end:
                return TreeNode(nums[start])

            mid = int((start + end) / 2)
            root = TreeNode(nums[mid])
            if mid == start:
                root.left = None
                root.right = buildTreeFromSortedArray(nums, mid + 1, end)
            else:
                root.left = buildTreeFromSortedArray(nums, start, mid - 1)
                root.right = buildTreeFromSortedArray(nums, mid + 1, end)
            return root

        if len(nums) == 0:
            return None
        return buildTreeFromSortedArray(nums, 0, len(nums) - 1)

    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        def getMiddleNode(head):
            fast = head
            slow = head
            if head is None:
                return None
            while slow.next and fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def constructBST(head):
            mid = getMiddleNode(head)
            root = TreeNode(mid.val)
            if head != mid:
                temp = head
                while temp != mid and temp.next != mid:
                    temp = temp.next
                temp.next = None
                root.left = constructBST(head)
            else:
                root.left = None
            if mid.next != None:
                root.right = constructBST(mid.next)
            else:
                root.right = None
            return root
        if not head:
            return None
        return constructBST(head)

    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max1, max2, max3 = -1000, -1000, -1000
        min1, min2, min3 = 1000, 1000, 1000
        for num in nums:
            if num > max1:
                max3 = max2
                max2 = max1
                max1 = num
            elif num > max2:
                max3 = max2
                max2 = num
            elif num > max3:
                max3 = num

            if num < min1:
                min3 = min2
                min2 = min1
                min1 = num
            elif num < min2:
                min3 = min2
                min2 = num
            elif num < min3:
                min3 = num

        return max(max1*max2*max3, min1*min2*max1)

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        lowestBuy = prices[0]
        maxProfit = 0
        for i in range(len(prices)):
            lowestBuy = min(prices[i], lowestBuy)
            maxProfit = max(maxProfit, prices[i] - lowestBuy)
        return maxProfit

    def maxProfit3(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) <= 1:
            return 0
        # leftMaxProfit
        lowestBuy = prices[0]
        leftMaxProfit = [0] * len(prices)
        for i in range(1, len(prices)):
            lowestBuy = min(prices[i], lowestBuy)
            leftMaxProfit[i] = max(leftMaxProfit[i-1], prices[i] - lowestBuy)
        # rightMaxProfit
        highestSell = prices[-1]
        rightMaxProfit = [0] * len(prices)
        for i in range(len(prices) - 2, -1, -1):
            highestSell = max(prices[i], highestSell)
            rightMaxProfit[i] = max(rightMaxProfit[i + 1], highestSell - prices[i])
        return max([leftMaxProfit[i] + rightMaxProfit[i] for i in range(len(prices))])

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.strip()
        def calculateSubstr2(s):
            index = len(s) - 1
            curStr = ''
            while index >= 0 and s[index] not in {'*', '/'}:
                if s[index] != ' ':
                    curStr = s[index] + curStr
                index -= 1
            if index == -1:
                return int(curStr)
            if s[index] == '*':
                return calculateSubstr2(s[0:index]) * calculateSubstr2(curStr)
            if s[index] == '/':
                return int(calculateSubstr2(s[0:index]) / calculateSubstr2(curStr))

        def calculateSubstr(s):
            index = len(s) - 1
            curStr = ''
            while index >= 0 and s[index] not in {'+', '-'}:
                if s[index] != ' ':
                    curStr = s[index] + curStr
                index -= 1
            if index == -1:
                return calculateSubstr2(curStr)
            if s[index] == '+':
                return calculateSubstr(s[0:index]) + calculateSubstr2(curStr)
            else:
                return calculateSubstr(s[0:index]) - calculateSubstr2(curStr)

        return calculateSubstr(s)

    def calculate2(self, s):
        if not s:
            return "0"
        stack, num, sign = [], 0, "+"
        for i in range(len(s)):
            if s[i].isdigit():
                num = num * 10 + ord(s[i]) - ord("0")
            if (not s[i].isdigit() and not s[i].isspace()) or i == len(s) - 1:
                if sign == "-":
                    stack.append(-num)
                elif sign == "+":
                    stack.append(num)
                elif sign == "*":
                    stack.append(stack.pop() * num)
                else:
                    tmp = stack.pop()
                    if tmp // num < 0 and tmp % num != 0:
                        stack.append(tmp // num + 1)
                    else:
                        stack.append(tmp // num)
                sign = s[i]
                num = 0
        return sum(stack)

    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        def hasPathSumRec(root, sum):
            leftPathSum = False
            rightPathSum = False
            if root.left is None and root.right is None:
                return root.val == sum
            if root.left is not None:
                leftPathSum = hasPathSumRec(root.left, sum - root.val)
            if root.right is not None:
                rightPathSum = hasPathSumRec(root.right, sum - root.val)
            return leftPathSum or rightPathSum
        if root is None:
            return False
        return hasPathSumRec(root, sum)

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        def pathSum(root, sum, curPath, res):
            curPath.append(root.val)
            if root.left is None and root.right is None:
                if root.val == sum:
                    res.append(curPath.copy())
                    return
                else:
                    return
            if root.left is not None:
                pathSum(root.left, sum - root.val, curPath, res)
                curPath.pop()
            if root.right is not None:
                pathSum(root.right, sum - root.val, curPath, res)
                curPath.pop()

        res = []
        curPath = []
        if root is None:
            return None
        pathSum(root, sum, curPath, res)
        return res

    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        def flattenRec(root):
            if root.left is None and root.right is None:
                return
            if root.left is None:
                flattenRec(root.right)
                return
            if root.right is None:
                flattenRec(root.left)
                root.right = root.left
                root.left = None
                return
            flattenRec(root.right)
            flattenRec(root.left)
            tempLeftRoot = root.left
            while tempLeftRoot.right:
                tempLeftRoot = tempLeftRoot.right
            tempLeftRoot.right = root.right
            root.right = root.left
            root.left = None
        if root is None:
            return
        flattenRec(root)

    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        def DFS(courseToCheck, prereqMap, courseOrder, courseChecked):
            if courseChecked[courseToCheck] == 1:
                return True
            if courseChecked[courseToCheck] == -1:
                return False

            courseChecked[courseToCheck] = -1
            if courseToCheck in prereqMap:
                for prereq in prereqMap[courseToCheck]:
                    if not DFS(prereq, prereqMap, courseOrder, courseChecked):
                        return False
            courseChecked[courseToCheck] = 1
            courseOrder.append(courseToCheck)
            return True
        if len(prerequisites) == 0:
            return range(0, numCourses)
        prereqMap = dict()
        for (course, prereq) in prerequisites:
            if course in prereqMap:
                prereqMap[course].append(prereq)
            else:
                prereqMap[course] = list()
                prereqMap[course].append(prereq)

        courseOrder = list()
        courseChecked = [0] * numCourses
        for courseToCheck in range(0, numCourses):
            # DFS is false if there is a circle
            if not DFS(courseToCheck, prereqMap, courseOrder, courseChecked):
                return list()
            courseChecked[courseToCheck] = 1
        return courseOrder

    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        curLvl = []
        res = []
        lvl = 0
        queue = [(lvl, root)]
        while len(queue) > 0:
            (nodelvl, node) = queue.pop()
            if nodelvl != lvl:
                lvl = nodelvl
                res.append(curLvl)
                curLvl = []
            if node.left is not None:
                queue.insert(0, (lvl + 1, node.left))
            if node.right is not None:
                queue.insert(0, (lvl + 1, node.right))
            curLvl.append(node.val)
        res.append(curLvl)
        return res

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        nrows = len(grid)
        ncols = len(grid[0])
        pathSum = [[0 for i in range(ncols)] for j in range(nrows)]
        for row in range(nrows - 1, -1, -1):
            for col in range(ncols - 1, -1, -1):
                if row == nrows -1 and col == ncols -1:
                    pathSum[row][col] = grid[row][col]
                elif col == ncols - 1:
                    pathSum[row][col] = pathSum[row+1][col] + grid[row][col]
                elif row == nrows - 1:
                    pathSum[row][col] = pathSum[row][col + 1] + grid[row][col]
                else:
                    pathSum[row][col] = min(pathSum[row][col + 1], pathSum[row + 1][col]) + \
                                        grid[row][col]
        return pathSum[0][0]

    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        nrows = len(triangle)
        prevLvlMinPathSum = triangle[0]
        if nrows == 1:
            return min(prevLvlMinPathSum)
        for i in range(1, nrows):
            curLvlMinPathSum = [0] * len(triangle[i])
            for j in range(0, len(curLvlMinPathSum)):
                if j == 0:
                    curLvlMinPathSum[j] = prevLvlMinPathSum[j] + triangle[i][j]
                elif j == len(curLvlMinPathSum) - 1:
                    curLvlMinPathSum[j] = prevLvlMinPathSum[-1] + triangle[i][j]
                else:
                    curLvlMinPathSum[j] = min(prevLvlMinPathSum[j-1],
                                              prevLvlMinPathSum[j]) + triangle[i][j]
            prevLvlMinPathSum = curLvlMinPathSum
        return min(curLvlMinPathSum)

    # buy interval: [[]]
    # sell interval: []
    # assume buy and sell intervals are both sorted by the start time
    def mergeIntvl(self, intvl1, intvl2):
        if intvl1[0] <= intvl2[0] and intvl1[1] >= intvl2[0]:
            res = [intvl1[0], max(intvl1[1], intvl2[1])]
        elif intvl2[0] <= intvl1[0] and intvl2[1] > intvl1[0]:
            res = [intvl2[0], max(intvl2[1], intvl1[1])]
        else:
            res = []
        return res

    def findActiveTimeOnBothSide(self, buyIntvls, sellIntvls):
        def mergeIntvl(intvl1, intvl2):
            if intvl1[0] <= intvl2[0] and intvl1[1] >= intvl2[0]:
                res = [intvl1[0], max(intvl1[1], intvl2[1])]
            elif intvl2[0] <= intvl1[0] and intvl2[1] > intvl1[0]:
                res = [intvl2[0], max(intvl2[1], intvl1[1])]
            else:
                res = []
            return res

        mergedBuyIntvls = []
        tmpBuyIntvl = [] if len(buyIntvls) == 0 else buyIntvls[0]
        for i in range(len(buyIntvls)):
            mergedIntvl = mergeIntvl(tmpBuyIntvl, buyIntvls[i])
            if not mergedIntvl:
                mergedBuyIntvls.append(tmpBuyIntvl)
                tmpBuyIntvl = buyIntvls[i]
            else:
                tmpBuyIntvl = mergedIntvl
        mergedBuyIntvls.append(tmpBuyIntvl)

        mergedSellIntvls = []
        tmpSellIntvl = [] if len(sellIntvls) == 0 else sellIntvls[0]
        for i in range(len(sellIntvls)):
            mergedIntvl = mergeIntvl(tmpSellIntvl, sellIntvls[i])
            if not mergedIntvl:
                mergedSellIntvls.append(tmpSellIntvl)
                tmpSellIntvl = sellIntvls[i]
            else:
                tmpSellIntvl = mergedIntvl
        mergedSellIntvls.append(tmpSellIntvl)

        buyPtr, sellPtr = 0, 0
        activeIntvls = []
        while buyPtr < len(mergedBuyIntvls) and sellPtr < len(mergedSellIntvls):
            buyIntvl = mergedBuyIntvls[buyPtr]
            sellIntvl = mergedSellIntvls[sellPtr]
            if buyIntvl[1] <= sellIntvl[0]:
                buyPtr += 1
            elif sellIntvl[1] <= buyIntvl[0]:
                sellPtr += 1
            else:
                intersectIntvl = [max(buyIntvl[0], sellIntvl[0]), min(buyIntvl[1], sellIntvl[1])]
                activeIntvls.append(intersectIntvl)
                if sellIntvl[1] > buyIntvl[1]:
                    buyPtr += 1
                else:
                    sellPtr += 1

        activeTime = 0
        for activeIntvl in activeIntvls:
            activeTime += activeIntvl[1] - activeIntvl[0]
        print(activeIntvls)
        return activeTime


def main():
    s = Solution()

    # print(s.mergeIntvl([1, 5], [0, 0.5]))
    print(s.findActiveTimeOnBothSide([[0, 10]], [[0, 7], [3, 3.5], [4, 5], [7, 9]]))
    print(s.findActiveTimeOnBothSide([[0, 3], [4, 9]], [[1, 2.5], [3.5, 7], [8, 10]]))

    # print(s.reverse(1534236469))
    # print(s.myAtoi("-   32"))
    # print(s.isPalindrome(21))
    # print(s.maxArea([2,3,4,5,18,17,6]))
    # print(s.intToRoman(3984))
    # print(s.longestCommonPrefix(["dog","racecar","car"]))
    # print(s.threeSum3([82597,-9243,62390,83030,-97960,-26521,-61011,83390,-38677,12333,75987,46091,83794,19355,-71037,-6242,-28801,324,1202,-90885,-2989,-95597,-34333,35528,5680,89093,-90606,50360,-29393,-27012,53313,65213,99818,-82405,-41661,-3333,-51952,72135,-1523,26377,74685,96992,92263,15929,5467,-99555,-43348,-41689,-60383,-3990,32165,65265,-72973,-58372,12741,-48568,-46596,72419,-1859,34153,62937,81310,-61823,-96770,-54944,8845,-91184,24208,-29078,31495,65258,14198,85395,70506,-40908,56740,-12228,-40072,32429,93001,68445,-73927,25731,-91859,-24150,10093,-60271,-81683,-18126,51055,48189,-6468,25057,81194,-58628,74042,66158,-14452,-49851,-43667,11092,39189,-17025,-79173,13606,83172,92647,-59741,19343,-26644,-57607,82908,-20655,1637,80060,98994,39331,-31274,-61523,91225,-72953,13211,-75116,-98421,-41571,-69074,99587,39345,42151,-2460,98236,15690,-52507,-95803,-48935,-46492,-45606,-79254,-99851,52533,73486,39948,-7240,71815,-585,-96252,90990,-93815,93340,-71848,58733,-14859,-83082,-75794,-82082,-24871,-15206,91207,-56469,-93618,67131,-8682,75719,87429,-98757,-7535,-24890,-94160,85003,33928,75538,97456,-66424,-60074,-8527,-28697,-22308,2246,-70134,-82319,-10184,87081,-34949,-28645,-47352,-83966,-60418,-15293,-53067,-25921,55172,75064,95859,48049,34311,-86931,-38586,33686,-36714,96922,76713,-22165,-80585,-34503,-44516,39217,-28457,47227,-94036,43457,24626,-87359,26898,-70819,30528,-32397,-69486,84912,-1187,-98986,-32958,4280,-79129,-65604,9344,58964,50584,71128,-55480,24986,15086,-62360,-42977,-49482,-77256,-36895,-74818,20,3063,-49426,28152,-97329,6086,86035,-88743,35241,44249,19927,-10660,89404,24179,-26621,-6511,57745,-28750,96340,-97160,-97822,-49979,52307,79462,94273,-24808,77104,9255,-83057,77655,21361,55956,-9096,48599,-40490,-55107,2689,29608,20497,66834,-34678,23553,-81400,-66630,-96321,-34499,-12957,-20564,25610,-4322,-58462,20801,53700,71527,24669,-54534,57879,-3221,33636,3900,97832,-27688,-98715,5992,24520,-55401,-57613,-69926,57377,-77610,20123,52174,860,60429,-91994,-62403,-6218,-90610,-37263,-15052,62069,-96465,44254,89892,-3406,19121,-41842,-87783,-64125,-56120,73904,-22797,-58118,-4866,5356,75318,46119,21276,-19246,-9241,-97425,57333,-15802,93149,25689,-5532,95716,39209,-87672,-29470,-16324,-15331,27632,-39454,56530,-16000,29853,46475,78242,-46602,83192,-73440,-15816,50964,-36601,89758,38375,-40007,-36675,-94030,67576,46811,-64919,45595,76530,40398,35845,41791,67697,-30439,-82944,63115,33447,-36046,-50122,-34789,43003,-78947,-38763,-89210,32756,-20389,-31358,-90526,-81607,88741,86643,98422,47389,-75189,13091,95993,-15501,94260,-25584,-1483,-67261,-70753,25160,89614,-90620,-48542,83889,-12388,-9642,-37043,-67663,28794,-8801,13621,12241,55379,84290,21692,-95906,-85617,-17341,-63767,80183,-4942,-51478,30997,-13658,8838,17452,-82869,-39897,68449,31964,98158,-49489,62283,-62209,-92792,-59342,55146,-38533,20496,62667,62593,36095,-12470,5453,-50451,74716,-17902,3302,-16760,-71642,-34819,96459,-72860,21638,47342,-69897,-40180,44466,76496,84659,13848,-91600,-90887,-63742,-2156,-84981,-99280,94326,-33854,92029,-50811,98711,-36459,-75555,79110,-88164,-97397,-84217,97457,64387,30513,-53190,-83215,252,2344,-27177,-92945,-89010,82662,-11670,86069,53417,42702,97082,3695,-14530,-46334,17910,77999,28009,-12374,15498,-46941,97088,-35030,95040,92095,-59469,-24761,46491,67357,-66658,37446,-65130,-50416,99197,30925,27308,54122,-44719,12582,-99525,-38446,-69050,-22352,94757,-56062,33684,-40199,-46399,96842,-50881,-22380,-65021,40582,53623,-76034,77018,-97074,-84838,-22953,-74205,79715,-33920,-35794,-91369,73421,-82492,63680,-14915,-33295,37145,76852,-69442,60125,-74166,74308,-1900,-30195,-16267,-60781,-27760,5852,38917,25742,-3765,49097,-63541,98612,-92865,-30248,9612,-8798,53262,95781,-42278,-36529,7252,-27394,-5021,59178,80934,-48480,-75131,-54439,-19145,-48140,98457,-6601,-51616,-89730,78028,32083,-48904,16822,-81153,-8832,48720,-80728,-45133,-86647,-4259,-40453,2590,28613,50523,-4105,-27790,-74579,-17223,63721,33489,-47921,97628,-97691,-14782,-65644,18008,-93651,-71266,80990,-76732,-47104,35368,28632,59818,-86269,-89753,34557,-92230,-5933,-3487,-73557,-13174,-43981,-43630,-55171,30254,-83710,-99583,-13500,71787,5017,-25117,-78586,86941,-3251,-23867,-36315,75973,86272,-45575,77462,-98836,-10859,70168,-32971,-38739,-12761,93410,14014,-30706,-77356,-85965,-62316,63918,-59914,-64088,1591,-10957,38004,15129,-83602,-51791,34381,-89382,-26056,8942,5465,71458,-73805,-87445,-19921,-80784,69150,-34168,28301,-68955,18041,6059,82342,9947,39795,44047,-57313,48569,81936,-2863,-80932,32976,-86454,-84207,33033,32867,9104,-16580,-25727,80157,-70169,53741,86522,84651,68480,84018,61932,7332,-61322,-69663,76370,41206,12326,-34689,17016,82975,-23386,39417,72793,44774,-96259,3213,79952,29265,-61492,-49337,14162,65886,3342,-41622,-62659,-90402,-24751,88511,54739,-21383,-40161,-96610,-24944,-602,-76842,-21856,69964,43994,-15121,-85530,12718,13170,-13547,69222,62417,-75305,-81446,-38786,-52075,-23110,97681,-82800,-53178,11474,35857,94197,-58148,-23689,32506,92154,-64536,-73930,-77138,97446,-83459,70963,22452,68472,-3728,-25059,-49405,95129,-6167,12808,99918,30113,-12641,-26665,86362,-33505,50661,26714,33701,89012,-91540,40517,-12716,-57185,-87230,29914,-59560,13200,-72723,58272,23913,-45586,-96593,-26265,-2141,31087,81399,92511,-34049,20577,2803,26003,8940,42117,40887,-82715,38269,40969,-50022,72088,21291,-67280,-16523,90535,18669,94342,-39568,-88080,-99486,-20716,23108,-28037,63342,36863,-29420,-44016,75135,73415,16059,-4899,86893,43136,-7041,33483,-67612,25327,40830,6184,61805,4247,81119,-22854,-26104,-63466,63093,-63685,60369,51023,51644,-16350,74438,-83514,99083,10079,-58451,-79621,48471,67131,-86940,99093,11855,-22272,-67683,-44371,9541,18123,37766,-70922,80385,-57513,-76021,-47890,36154,72935,84387,-92681,-88303,-7810,59902,-90,-64704,-28396,-66403,8860,13343,33882,85680,7228,28160,-14003,54369,-58893,92606,-63492,-10101,64714,58486,29948,-44679,-22763,10151,-56695,4031,-18242,-36232,86168,-14263,9883,47124,47271,92761,-24958,-73263,-79661,-69147,-18874,29546,-92588,-85771,26451,-86650,-43306,-59094,-47492,-34821,-91763,-47670,33537,22843,67417,-759,92159,63075,94065,-26988,55276,65903,30414,-67129,-99508,-83092,-91493,-50426,14349,-83216,-76090,32742,-5306,-93310,-60750,-60620,-45484,-21108,-58341,-28048,-52803,69735,78906,81649,32565,-86804,-83202,-65688,-1760,89707,93322,-72750,84134,71900,-37720,19450,-78018,22001,-23604,26276,-21498,65892,-72117,-89834,-23867,55817,-77963,42518,93123,-83916,63260,-2243,-97108,85442,-36775,17984,-58810,99664,-19082,93075,-69329,87061,79713,16296,70996,13483,-74582,49900,-27669,-40562,1209,-20572,34660,83193,75579,7344,64925,88361,60969,3114,44611,-27445,53049,-16085,-92851,-53306,13859,-33532,86622,-75666,-18159,-98256,51875,-42251,-27977,-18080,23772,38160,41779,9147,94175,99905,-85755,62535,-88412,-52038,-68171,93255,-44684,-11242,-104,31796,62346,-54931,-55790,-70032,46221,56541,-91947,90592,93503,4071,20646,4856,-63598,15396,-50708,32138,-85164,38528,-89959,53852,57915,-42421,-88916,-75072,67030,-29066,49542,-71591,61708,-53985,-43051,28483,46991,-83216,80991,-46254,-48716,39356,-8270,-47763,-34410,874,-1186,-7049,28846,11276,21960,-13304,-11433,-4913,55754,79616,70423,-27523,64803,49277,14906,-97401,-92390,91075,70736,21971,-3303,55333,-93996,76538,54603,-75899,98801,46887,35041,48302,-52318,55439,24574,14079,-24889,83440,14961,34312,-89260,-22293,-81271,-2586,-71059,-10640,-93095,-5453,-70041,66543,74012,-11662,-52477,-37597,-70919,92971,-17452,-67306,-80418,7225,-89296,24296,86547,37154,-10696,74436,-63959,58860,33590,-88925,-97814,-83664,85484,-8385,-50879,57729,-74728,-87852,-15524,-91120,22062,28134,80917,32026,49707,-54252,-44319,-35139,13777,44660,85274,25043,58781,-89035,-76274,6364,-63625,72855,43242,-35033,12820,-27460,77372,-47578,-61162,-70758,-1343,-4159,64935,56024,-2151,43770,19758,-30186,-86040,24666,-62332,-67542,73180,-25821,-27826,-45504,-36858,-12041,20017,-24066,-56625,-52097,-47239,-90694,8959,7712,-14258,-5860,55349,61808,-4423,-93703,64681,-98641,-25222,46999,-83831,-54714,19997,-68477,66073,51801,-66491,52061,-52866,79907,-39736,-68331,68937,91464,98892,910,93501,31295,-85873,27036,-57340,50412,21,-2445,29471,71317,82093,-94823,-54458,-97410,39560,-7628,66452,39701,54029,37906,46773,58296,60370,-61090,85501,-86874,71443,-72702,-72047,14848,34102,77975,-66294,-36576,31349,52493,-70833,-80287,94435,39745,-98291,84524,-18942,10236,93448,50846,94023,-6939,47999,14740,30165,81048,84935,-19177,-13594,32289,62628,-90612,-542,-66627,64255,71199,-83841,-82943,-73885,8623,-67214,-9474,-35249,62254,-14087,-90969,21515,-83303,94377,-91619,19956,-98810,96727,-91939,29119,-85473,-82153,-69008,44850,74299,-76459,-86464,8315,-49912,-28665,59052,-69708,76024,-92738,50098,18683,-91438,18096,-19335,35659,91826,15779,-73070,67873,-12458,-71440,-46721,54856,97212,-81875,35805,36952,68498,81627,-34231,81712,27100,-9741,-82612,18766,-36392,2759,41728,69743,26825,48355,-17790,17165,56558,3295,-24375,55669,-16109,24079,73414,48990,-11931,-78214,90745,19878,35673,-15317,-89086,94675,-92513,88410,-93248,-19475,-74041,-19165,32329,-26266,-46828,-18747,45328,8990,-78219,-25874,-74801,-44956,-54577,-29756,-99822,-35731,-18348,-68915,-83518,-53451,95471,-2954,-13706,-8763,-21642,-37210,16814,-60070,-42743,27697,-36333,-42362,11576,85742,-82536,68767,-56103,-63012,71396,-78464,-68101,-15917,-11113,-3596,77626,-60191,-30585,-73584,6214,-84303,18403,23618,-15619,-89755,-59515,-59103,-74308,-63725,-29364,-52376,-96130,70894,-12609,50845,-2314,42264,-70825,64481,55752,4460,-68603,-88701,4713,-50441,-51333,-77907,97412,-66616,-49430,60489,-85262,-97621,-18980,44727,-69321,-57730,66287,-92566,-64427,-14270,11515,-92612,-87645,61557,24197,-81923,-39831,-10301,-23640,-76219,-68025,92761,-76493,68554,-77734,-95620,-11753,-51700,98234,-68544,-61838,29467,46603,-18221,-35441,74537,40327,-58293,75755,-57301,-7532,-94163,18179,-14388,-22258,-46417,-48285,18242,-77551,82620,250,-20060,-79568,-77259,82052,-98897,-75464,48773,-79040,-11293,45941,-67876,-69204,-46477,-46107,792,60546,-34573,-12879,-94562,20356,-48004,-62429,96242,40594,2099,99494,25724,-39394,-2388,-18563,-56510,-83570,-29214,3015,74454,74197,76678,-46597,60630,-76093,37578,-82045,-24077,62082,-87787,-74936,58687,12200,-98952,70155,-77370,21710,-84625,-60556,-84128,925,65474,-15741,-94619,88377,89334,44749,22002,-45750,-93081,-14600,-83447,46691,85040,-66447,-80085,56308,44310,24979,-29694,57991,4675,-71273,-44508,13615,-54710,23552,-78253,-34637,50497,68706,81543,-88408,-21405,6001,-33834,-21570,-46692,-25344,20310,71258,-97680,11721,59977,59247,-48949,98955,-50276,-80844,-27935,-76102,55858,-33492,40680,66691,-33188,8284,64893,-7528,6019,-85523,8434,-64366,-56663,26862,30008,-7611,-12179,-70076,21426,-11261,-36864,-61937,-59677,929,-21052,3848,-20888,-16065,98995,-32293,-86121,-54564,77831,68602,74977,31658,40699,29755,98424,80358,-69337,26339,13213,-46016,-18331,64713,-46883,-58451,-70024,-92393,-4088,70628,-51185,71164,-75791,-1636,-29102,-16929,-87650,-84589,-24229,-42137,-15653,94825,13042,88499,-47100,-90358,-7180,29754,-65727,-42659,-85560,-9037,-52459,20997,-47425,17318,21122,20472,-23037,65216,-63625,-7877,-91907,24100,-72516,22903,-85247,-8938,73878,54953,87480,-31466,-99524,35369,-78376,89984,-15982,94045,-7269,23319,-80456,-37653,-76756,2909,81936,54958,-12393,60560,-84664,-82413,66941,-26573,-97532,64460,18593,-85789,-38820,-92575,-43663,-89435,83272,-50585,13616,-71541,-53156,727,-27644,16538,34049,57745,34348,35009,16634,-18791,23271,-63844,95817,21781,16590,59669,15966,-6864,48050,-36143,97427,-59390,96931,78939,-1958,50777,43338,-51149,39235,-27054,-43492,67457,-83616,37179,10390,85818,2391,73635,87579,-49127,-81264,-79023,-81590,53554,-74972,-83940,-13726,-39095,29174,78072,76104,47778,25797,-29515,-6493,-92793,22481,-36197,-65560,42342,15750,97556,99634,-56048,-35688,13501,63969,-74291,50911,39225,93702,-3490,-59461,-30105,-46761,-80113,92906,-68487,50742,36152,-90240,-83631,24597,-50566,-15477,18470,77038,40223,-80364,-98676,70957,-63647,99537,13041,31679,86631,37633,-16866,13686,-71565,21652,-46053,-80578,-61382,68487,-6417,4656,20811,67013,-30868,-11219,46,74944,14627,56965,42275,-52480,52162,-84883,-52579,-90331,92792,42184,-73422,-58440,65308,-25069,5475,-57996,59557,-17561,2826,-56939,14996,-94855,-53707,99159,43645,-67719,-1331,21412,41704,31612,32622,1919,-69333,-69828,22422,-78842,57896,-17363,27979,-76897,35008,46482,-75289,65799,20057,7170,41326,-76069,90840,-81253,-50749,3649,-42315,45238,-33924,62101,96906,58884,-7617,-28689,-66578,62458,50876,-57553,6739,41014,-64040,-34916,37940,13048,-97478,-11318,-89440,-31933,-40357,-59737,-76718,-14104,-31774,28001,4103,41702,-25120,-31654,63085,-3642,84870,-83896,-76422,-61520,12900,88678,85547,33132,-88627,52820,63915,-27472,78867,-51439,33005,-23447,-3271,-39308,39726,-74260,-31874,-36893,93656,910,-98362,60450,-88048,99308,13947,83996,-90415,-35117,70858,-55332,-31721,97528,82982,-86218,6822,25227,36946,97077,-4257,-41526,56795,89870,75860,-70802,21779,14184,-16511,-89156,-31422,71470,69600,-78498,74079,-19410,40311,28501,26397,-67574,-32518,68510,38615,19355,-6088,-97159,-29255,-92523,3023,-42536,-88681,64255,41206,44119,52208,39522,-52108,91276,-70514,83436,63289,-79741,9623,99559,12642,85950,83735,-21156,-67208,98088,-7341,-27763,-30048,-44099,-14866,-45504,-91704,19369,13700,10481,-49344,-85686,33994,19672,36028,60842,66564,-24919,33950,-93616,-47430,-35391,-28279,56806,74690,39284,-96683,-7642,-75232,37657,-14531,-86870,-9274,-26173,98640,88652,64257,46457,37814,-19370,9337,-22556,-41525,39105,-28719,51611,-93252,98044,-90996,21710,-47605,-64259,-32727,53611,-31918,-3555,33316,-66472,21274,-37731,-2919,15016,48779,-88868,1897,41728,46344,-89667,37848,68092,-44011,85354,-43776,38739,-31423,-66330,65167,-22016,59405,34328,-60042,87660,-67698,-59174,-1408,-46809,-43485,-88807,-60489,13974,22319,55836,-62995,-37375,-4185,32687,-36551,-75237,58280,26942,-73756,71756,78775,-40573,14367,-71622,-77338,24112,23414,-7679,-51721,87492,85066,-21612,57045,10673,-96836,52461,-62218,-9310,65862,-22748,89906,-96987,-98698,26956,-43428,46141,47456,28095,55952,67323,-36455,-60202,-43302,-82932,42020,77036,10142,60406,70331,63836,58850,-66752,52109,21395,-10238,-98647,-41962,27778,69060,98535,-28680,-52263,-56679,66103,-42426,27203,80021,10153,58678,36398,63112,34911,20515,62082,-15659,-40785,27054,43767,-20289,65838,-6954,-60228,-72226,52236,-35464,25209,-15462,-79617,-41668,-84083,62404,-69062,18913,46545,20757,13805,24717,-18461,-47009,-25779,68834,64824,34473,39576,31570,14861,-15114,-41233,95509,68232,67846,84902,-83060,17642,-18422,73688,77671,-26930,64484,-99637,73875,6428,21034,-73471,19664,-68031,15922,-27028,48137,54955,-82793,-41144,-10218,-24921,-28299,-2288,68518,-54452,15686,-41814,66165,-72207,-61986,80020,50544,-99500,16244,78998,40989,14525,-56061,-24692,-94790,21111,37296,-90794,72100,70550,-31757,17708,-74290,61910,78039,-78629,-25033,73172,-91953,10052,64502,99585,-1741,90324,-73723,68942,28149,30218,24422,16659,10710,-62594,94249,96588,46192,34251,73500,-65995,-81168,41412,-98724,-63710,-54696,-52407,19746,45869,27821,-94866,-76705,-13417,-61995,-71560,43450,67384,-8838,-80293,-28937,23330,-89694,-40586,46918,80429,-5475,78013,25309,-34162,37236,-77577,86744,26281,-29033,-91813,35347,13033,-13631,-24459,3325,-71078,-75359,81311,19700,47678,-74680,-84113,45192,35502,37675,19553,76522,-51098,-18211,89717,4508,-82946,27749,85995,89912,-53678,-64727,-14778,32075,-63412,-40524,86440,-2707,-36821,63850,-30883,67294,-99468,-23708,34932,34386,98899,29239,-23385,5897,54882,98660,49098,70275,17718,88533,52161,63340,50061,-89457,19491,-99156,24873,-17008,64610,-55543,50495,17056,-10400,-56678,-29073,-42960,-76418,98562,-88104,-96255,10159,-90724,54011,12052,45871,-90933,-69420,67039,37202,78051,-52197,-40278,-58425,65414,-23394,-1415,6912,-53447,7352,17307,-78147,63727,98905,55412,-57658,-32884,-44878,22755,39730,3638,35111,39777,74193,38736,-11829,-61188,-92757,55946,-71232,-63032,-83947,39147,-96684,-99233,25131,-32197,24406,-55428,-61941,25874,-69453,64483,-19644,-68441,12783,87338,-48676,66451,-447,-61590,50932,-11270,29035,65698,-63544,10029,80499,-9461,86368,91365,-81810,-71914,-52056,-13782,44240,-30093,-2437,24007,67581,-17365,-69164,-8420,-69289,-29370,48010,90439,13141,69243,50668,39328,61731,78266,-81313,17921,-38196,55261,9948,-24970,75712,-72106,28696,7461,31621,61047,51476,56512,11839,-96916,-82739,28924,-99927,58449,37280,69357,11219,-32119,-62050,-48745,-83486,-52376,42668,82659,68882,38773,46269,-96005,97630,25009,-2951,-67811,99801,81587,-79793,-18547,-83086,69512,33127,-92145,-88497,47703,59527,1909,88785,-88882,69188,-46131,-5589,-15086,36255,-53238,-33009,82664,53901,35939,-42946,-25571,33298,69291,53199,74746,-40127,-39050,91033,51717,-98048,87240,36172,65453,-94425,-63694,-30027,59004,88660,3649,-20267,-52565,-67321,34037,4320,91515,-56753,60115,27134,68617,-61395,-26503,-98929,-8849,-63318,10709,-16151,61905,-95785,5262,23670,-25277,90206,-19391,45735,37208,-31992,-92450,18516,-90452,-58870,-58602,93383,14333,17994,82411,-54126,-32576,35440,-60526,-78764,-25069,-9022,-394,92186,-38057,55328,-61569,67780,77169,19546,-92664,-94948,44484,-13439,83529,27518,-48333,72998,38342,-90553,-98578,-76906,81515,-16464,78439,92529,35225,-39968,-10130,-7845,-32245,-74955,-74996,67731,-13897,-82493,33407,93619,59560,-24404,-57553,19486,-45341,34098,-24978,-33612,79058,71847,76713,-95422,6421,-96075,-59130,-28976,-16922,-62203,69970,68331,21874,40551,89650,51908,58181,66480,-68177,34323,-3046,-49656,-59758,43564,-10960,-30796,15473,-20216,46085,-85355,41515,-30669,-87498,57711,56067,63199,-83805,62042,91213,-14606,4394,-562,74913,10406,96810,-61595,32564,31640,-9732,42058,98052,-7908,-72330,1558,-80301,34878,32900,3939,-8824,88316,20937,21566,-3218,-66080,-31620,86859,54289,90476,-42889,-15016,-18838,75456,30159,-67101,42328,-92703,85850,-5475,23470,-80806,68206,17764,88235,46421,-41578,74005,-81142,80545,20868,-1560,64017,83784,68863,-97516,-13016,-72223,79630,-55692,82255,88467,28007,-34686,-69049,-41677,88535,-8217,68060,-51280,28971,49088,49235,26905,-81117,-44888,40623,74337,-24662,97476,79542,-72082,-35093,98175,-61761,-68169,59697,-62542,-72965,59883,-64026,-37656,-92392,-12113,-73495,98258,68379,-21545,64607,-70957,-92254,-97460,-63436,-8853,-19357,-51965,-76582,12687,-49712,45413,-60043,33496,31539,-57347,41837,67280,-68813,52088,-13155,-86430,-15239,-45030,96041,18749,-23992,46048,35243,-79450,85425,-58524,88781,-39454,53073,-48864,-82289,39086,82540,-11555,25014,-5431,-39585,-89526,2705,31953,-81611,36985,-56022,68684,-27101,11422,64655,-26965,-63081,-13840,-91003,-78147,-8966,41488,1988,99021,-61575,-47060,65260,-23844,-21781,-91865,-19607,44808,2890,63692,-88663,-58272,15970,-65195,-45416,-48444,-78226,-65332,-24568,42833,-1806,-71595,80002,-52250,30952,48452,-90106,31015,-22073,62339,63318,78391,28699,77900,-4026,-76870,-45943,33665,9174,-84360,-22684,-16832,-67949,-38077,-38987,-32847,51443,-53580,-13505,9344,-92337,26585,70458,-52764,-67471,-68411,-1119,-2072,-93476,67981,40887,-89304,-12235,41488,1454,5355,-34855,-72080,24514,-58305,3340,34331,8731,77451,-64983,-57876,82874,62481,-32754,-39902,22451,-79095,-23904,78409,-7418,77916]))
    # print(s.threeSumClosest([-1, 2, 1, -4], 1))
    # print(s.letterCombinations("29"))
    # print(s.fourSum2([-495,-482,-455,-447,-400,-388,-381,-360,-350,-340,-330,-317,-308,-300,-279,-235,-209,-206,-188,-186,-171,-145,-143,-141,-137,-129,-121,-115,-97,-56,-47,-28,-20,-19,8,11,35,41,46,50,69,84,85,86,88,91,135,160,171,172,177,190,226,234,238,244,249,253,254,272,281,284,294,296,300,303,307,313,320,320,327,334,355,362,367,401,426,436,456,467,473,473,484], -7178))
    # print(s.fourSum2([0, 0, 0, 0], 0))
    # myList = ListNode(1)
    # myList.insertNode(2)
    # myList.insertNode(3)
    # myList.insertNode(4)
    # myList.insertNode(5)
    # myList.insertNode(6)
    # myList.insertNode(7)
    # myList.printList()
    # print(s.isValid("[{}]()"))
    # print(s.generateParenthesis(3))
    # nums = [1, 1, 2, 2, 3, 3, 4, 6]
    # # Remove duplicates
    # print("Remove duplicates: ")
    # print(s.removeDuplicates(nums))
    # print(nums)
    # myList = s.swapPairs(myList)
    # myList.printList()
    # nums1 = [1, 2, 0, 0, 0, 0]
    # nums2 = [2, 5, 6]
    # s.merge(nums1, 2, nums2, 3)
    # list1 = ListNode(1)
    # list1.insertNode(4)
    # list1.insertNode(5)
    # list2 = ListNode(1)
    # list2.insertNode(3)
    # list2.insertNode(4)
    # list3 = ListNode(2)
    # list3.insertNode(6)

    # result = s.mergeKLists([list1, list2, list3])
    # result.printList()

    # reversedList = s.reverseLinkedList(myList)
    # reversedList.printList()
    # kreversedList = s.reverseKGroup(myList, 1)
    # kreversedList.printList()
    # mylen = s.removeElement([3], 3)
    # print(mylen)

    # quotient = s.divide(2147483647, 3)
    # print(quotient)
    # a = s.fib(100)
    # print(s.findSubset(16, [2, 4, 6, 10]))
    # a = [3, 1, 2, 4, 5]
    # s.nextPermutation(a)
    # parens = ")()())()()("
    # print(s.longestValidParentheses(parens))
    # print(s.search([4, 5, 6, 7, 8, 1, 2, 3], 8))
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(2)
#     root.left.left = TreeNode(3)
#     root.left.right = TreeNode(3)
#     root.left.left.left = TreeNode(4)
#     root.left.left.right = TreeNode(4)
#
#     print(s.levelOrderBottom(root))
#     print(s.minDepth(root))
#     print(s.rightSideView2(root))
#     print(s.zigzagLevelOrder(root))
#
#     a = [[0, 0, 0], [0, 1, 0], [1, 1, 1]]
#     for i in range(0, len(a)):
#         print(a[i])
#     b = s.updateMatrix(a)
#     for i in range(0, len(b)):
#         print(b[i])
#
#     nums = [5, 2, 3, 7, 7, 1, 9, 2, 4]
#     print(s.mergeSort(nums))
#     print(s.isSymmetric(root))
#
#     my_list = ListNode(5)
#     my_list.insertNode(2)
#     my_list.insertNode(3)
#     my_list.insertNode(7)
#     my_list.insertNode(7)
#     my_list.insertNode(1)
#     my_list.insertNode(9)
#     my_list.insertNode(2)
#     my_list.insertNode(4)
#     my_list.printList()
#
#     my_sortedList = s.sortList(my_list)
#     my_sortedList.printList()
#
#     print(s.isBalanced(root))
#     print(s.generate(5))
#
# #     prerequisites = \
# # [[785,230],[843,838],[725,91],[236,135],[804,544],[779,204],[599,306],[685,651],[716,562],[419,381],[575,549],[895,348],[872,16],[938,344],[565,340],[794,21],[867,557],[857,486],[256,131],[959,439],[756,728],[873,330],[320,99],[825,657],[620,63],[534,404],[795,385],[171,159],[982,854],[458,203],[243,107],[403,289],[868,400],[313,214],[851,368],[773,767],[276,0],[948,672],[439,100],[437,255],[272,175],[758,158],[495,453],[480,158],[240,61],[970,568],[221,215],[758,22],[310,106],[822,111],[229,163],[386,150],[293,94],[950,25],[959,680],[858,78],[819,512],[672,385],[830,353],[961,919],[757,507],[180,98],[755,237],[382,308],[502,260],[987,407],[834,646],[963,895],[348,320],[973,436],[96,32],[916,857],[373,287],[948,205],[277,84],[467,386],[663,289],[763,152],[788,323],[958,514],[757,675],[980,387],[494,78],[883,245],[974,615],[467,153],[763,40],[732,626],[355,244],[751,586],[839,11],[675,13],[52,19],[853,6],[758,296],[534,339],[898,550],[744,59],[822,166],[338,20],[730,149],[979,725],[539,188],[848,413],[798,115],[399,215],[832,268],[709,319],[894,175],[998,373],[858,480],[263,189],[522,25],[613,377],[736,602],[253,94],[375,128],[374,54],[929,463],[722,265],[435,261],[841,780],[585,324],[46,14],[972,142],[811,715],[514,142],[240,142],[423,412],[856,140],[643,149],[570,399],[491,390],[498,303],[919,514],[616,488],[497,447],[235,179],[362,83],[913,323],[767,502],[470,336],[398,2],[702,16],[420,3],[670,74],[942,107],[579,513],[466,252],[775,608],[321,251],[653,22],[955,743],[923,64],[443,48],[817,152],[288,180],[983,660],[223,85],[816,341],[812,247],[733,264],[610,204],[761,400],[652,592],[840,39],[929,325],[814,203],[477,373],[888,337],[722,92],[980,260],[532,181],[886,639],[784,421],[962,531],[784,57],[423,30],[277,172],[558,349],[781,403],[998,216],[864,741],[941,901],[996,413],[944,400],[994,394],[781,612],[607,206],[453,261],[462,160],[534,406],[637,4],[897,713],[867,244],[463,33],[473,462],[149,35],[78,39],[479,91],[832,280],[923,717],[447,147],[254,141],[332,121],[698,196],[576,233],[914,272],[891,149],[447,140],[719,449],[343,274],[935,268],[495,156],[433,59],[969,526],[752,347],[948,291],[734,357],[795,218],[274,47],[863,32],[427,289],[969,859],[453,46],[428,212],[702,152],[773,435],[539,504],[318,14],[733,141],[307,268],[499,482],[713,189],[706,325],[490,349],[970,120],[892,72],[942,484],[771,718],[784,325],[665,543],[530,474],[827,90],[564,360],[247,172],[610,374],[853,700],[364,233],[893,425],[695,530],[382,42],[779,582],[472,17],[988,543],[498,214],[895,532],[851,165],[960,845],[825,470],[626,373],[891,186],[830,658],[986,925],[541,210],[633,606],[847,634],[538,454],[673,25],[417,140],[774,42],[275,143],[608,495],[736,274],[687,403],[882,434],[729,345],[773,502],[955,184],[932,127],[577,524],[705,289],[918,28],[506,441],[663,352],[566,530],[256,255],[206,64],[511,256],[771,393],[732,279],[489,464],[727,451],[653,505],[921,749],[915,537],[906,90],[931,420],[970,543],[624,537],[354,126],[554,373],[699,476],[901,109],[600,388],[179,81],[829,205],[824,549],[987,861],[578,538],[356,147],[666,287],[932,264],[392,138],[779,98],[715,320],[564,509],[646,293],[294,53],[706,472],[548,512],[905,860],[926,804],[715,323],[788,547],[655,419],[813,451],[528,482],[779,93],[908,193],[463,200],[847,284],[231,128],[620,361],[372,169],[435,257],[394,268],[420,368],[850,130],[631,144],[657,63],[423,222],[580,116],[382,334],[385,242],[265,125],[325,125],[209,71],[519,274],[917,870],[779,742],[751,604],[839,281],[483,287],[256,34],[666,389],[913,135],[513,161],[666,170],[426,425],[804,631],[461,280],[507,156],[758,73],[764,306],[905,499],[234,86],[630,252],[876,124],[318,275],[331,301],[874,190],[969,681],[862,302],[885,794],[616,206],[699,142],[877,538],[802,475],[704,424],[479,300],[916,326],[932,573],[845,230],[224,46],[790,401],[795,700],[639,176],[917,471],[652,217],[944,928],[518,110],[661,103],[874,186],[206,72],[730,404],[969,674],[801,770],[535,1],[390,243],[270,211],[626,545],[875,464],[601,484],[630,505],[195,82],[891,703],[545,411],[603,252],[215,177],[729,440],[951,351],[476,42],[876,261],[812,248],[735,110],[386,102],[534,72],[556,310],[326,282],[608,226],[900,17],[667,555],[794,164],[709,304],[907,297],[421,62],[608,102],[289,58],[383,84],[895,254],[694,156],[968,311],[932,139],[657,67],[936,659],[875,759],[483,233],[490,469],[84,40],[321,225],[843,341],[449,255],[719,325],[854,271],[697,47],[594,320],[734,253],[655,552],[302,154],[704,624],[674,308],[790,609],[883,427],[549,417],[972,196],[933,119],[999,40],[760,745],[543,84],[994,19],[506,343],[440,121],[947,54],[713,289],[642,210],[810,410],[820,808],[964,684],[963,709],[347,292],[973,503],[943,204],[728,577],[851,741],[549,215],[415,40],[957,439],[279,181],[931,883],[840,84],[742,555],[803,196],[884,332],[708,352],[643,192],[976,278],[693,665],[661,462],[792,538],[961,238],[133,107],[561,180],[860,508],[924,616],[870,660],[776,369],[793,307],[944,868],[856,56],[415,376],[444,397],[628,426],[790,45],[304,108],[457,349],[441,161],[515,94],[899,109],[867,195],[718,404],[619,462],[103,33],[862,77],[727,672],[825,815],[722,556],[934,883],[828,322],[630,368],[282,170],[824,371],[851,779],[691,155],[533,382],[884,61],[697,130],[201,105],[798,296],[855,265],[503,31],[388,124],[850,152],[933,203],[736,80],[335,239],[446,6],[630,178],[802,88],[299,270],[696,526],[869,269],[659,292],[284,93],[719,212],[710,480],[497,205],[565,123],[547,442],[582,385],[670,550],[729,218],[731,285],[405,57],[734,683],[424,102],[823,39],[665,629],[392,24],[802,242],[791,620],[864,112],[178,94],[898,511],[978,595],[887,3],[521,144],[515,178],[908,302],[233,60],[735,161],[347,241],[736,117],[233,192],[633,449],[831,806],[308,304],[868,425],[307,33],[907,67],[501,265],[945,62],[516,88],[922,119],[358,150],[969,680],[787,500],[400,366],[662,569],[767,316],[821,38],[945,13],[871,413],[536,251],[647,591],[645,84],[418,242],[153,37],[647,32],[867,298],[667,203],[988,398],[890,578],[988,712],[454,369],[420,92],[644,631],[458,215],[921,153],[913,490],[947,594],[774,121],[286,212],[489,8],[289,236],[809,664],[793,694],[940,395],[647,441],[953,352],[521,194],[924,592],[885,842],[199,149],[489,243],[736,692],[424,329],[788,9],[606,457],[685,615],[405,403],[604,151],[553,430],[876,372],[920,598],[939,666],[993,109],[708,115],[760,153],[753,455],[125,13],[688,153],[883,161],[769,427],[960,308],[824,817],[479,169],[829,163],[773,280],[735,612],[938,610],[914,378],[832,126],[920,715],[653,429],[849,412],[542,103],[668,47],[917,603],[504,271],[760,399],[751,7],[945,327],[46,19],[316,176],[411,38],[666,136],[818,384],[932,108],[972,658],[666,23],[868,518],[690,258],[241,226],[553,535],[837,269],[988,619],[983,285],[635,259],[800,110],[573,65],[841,318],[531,424],[705,77],[474,154],[757,552],[606,18],[871,735],[961,623],[605,13],[900,121],[180,7],[906,737],[862,495],[582,443],[233,197],[629,430],[370,32],[890,864],[879,659],[778,248],[988,434],[521,109],[903,668],[891,809],[529,247],[794,137],[663,648],[988,376],[924,712],[640,319],[409,348],[780,447],[918,327],[570,103],[473,203],[743,469],[445,282],[965,926],[720,359],[862,520],[772,130],[319,192],[703,75],[954,172],[885,208],[629,484],[775,587],[919,260],[835,395],[707,681],[826,105],[991,840],[638,409],[965,125],[949,289],[809,408],[923,423],[853,340],[781,748],[519,0],[317,172],[723,49],[909,580],[780,62],[830,789],[813,784],[903,249],[570,440],[631,625],[720,32],[381,122],[406,112],[865,620],[736,497],[730,280],[417,35],[638,40],[767,531],[679,147],[485,19],[905,554],[897,651],[491,139],[731,722],[627,516],[486,107],[766,184],[793,223],[857,489],[576,156],[878,784],[979,892],[611,501],[964,651],[367,231],[183,59],[905,242],[487,45],[912,534],[623,367],[901,692],[882,816],[407,275],[788,230],[887,286],[652,341],[915,73],[303,45],[960,304],[705,316],[707,314],[920,351],[956,48],[818,814],[529,349],[507,236],[764,206],[911,217],[429,323],[903,487],[642,376],[763,329],[175,66],[823,171],[856,395],[590,106],[384,323],[616,561],[792,785],[887,387],[823,431],[521,418],[161,101],[929,367],[612,352],[859,549],[810,147],[314,13],[920,71],[783,366],[525,399],[300,19],[653,593],[165,99],[772,349],[804,209],[926,156],[335,241],[465,294],[567,502],[365,167],[689,651],[684,403],[584,1],[902,283],[630,311],[743,156],[666,660],[114,90],[542,237],[606,229],[609,206],[514,220],[931,650],[928,398],[892,513],[591,276],[685,406],[185,88],[566,357],[875,495],[798,417],[286,13],[708,410],[748,375],[970,403],[966,910],[943,514],[678,433],[933,876],[489,423],[337,143],[710,491],[932,59],[972,907],[399,174],[396,44],[349,298],[355,230],[719,608],[682,113],[333,172],[930,783],[916,344],[514,92],[614,153],[676,218],[442,425],[556,266],[851,381],[683,247],[285,83],[939,556],[221,35],[546,109],[183,144],[827,506],[737,64],[488,433],[756,294],[842,17],[839,37],[881,256],[689,317],[974,592],[488,225],[603,342],[649,301],[641,478],[904,574],[437,57],[898,59],[509,478],[479,144],[78,16],[231,4],[677,298],[943,58],[632,14],[699,281],[852,308],[251,211],[448,333],[361,289],[837,397],[380,60],[848,340],[981,531],[219,83],[301,112],[993,36],[948,22],[843,830],[858,358],[491,392],[696,45],[983,664],[886,456],[557,521],[919,674],[180,36],[638,291],[951,661],[597,283],[950,881],[515,288],[531,418],[514,499],[254,102],[350,215],[509,25],[610,71],[533,18],[453,42],[157,120],[626,434],[800,671],[900,211],[829,626],[699,269],[415,343],[677,116],[521,82],[568,8],[492,454],[318,197],[645,131],[675,360],[795,563],[254,129],[793,149],[709,512],[591,138],[411,148],[127,25],[154,9],[470,87],[852,660],[739,60],[875,280],[701,332],[288,234],[923,74],[909,362],[870,330],[869,647],[883,338],[711,246],[196,30],[716,393],[811,201],[970,808],[607,398],[755,732],[528,431],[936,237],[633,406],[950,922],[129,53],[904,891],[633,589],[470,129],[432,386],[688,35],[798,663],[319,202],[791,335],[425,63],[937,570],[170,19],[298,262],[714,194],[827,797],[913,824],[584,483],[738,234],[448,161],[942,931],[495,485],[810,683],[885,228],[213,159],[983,716],[611,172],[830,138],[389,332],[970,166],[251,21],[669,555],[439,33],[381,204],[881,724],[843,842],[640,449],[183,32],[384,66],[557,305],[462,424],[841,676],[888,39],[777,511],[994,798],[613,81],[922,574],[442,330],[185,50],[976,48],[428,119],[994,863],[473,182],[786,506],[807,497],[668,276],[308,195],[644,65],[632,188],[673,338],[402,300],[894,336],[420,96],[732,318],[475,402],[969,636],[728,487],[370,163],[648,638],[997,674],[929,386],[341,167],[890,564],[732,368],[711,644],[876,834],[152,150],[139,138],[834,185],[570,185],[966,26],[816,101],[880,642],[602,92],[907,463],[435,367],[405,154],[256,20],[820,36],[411,194],[103,95],[929,59],[648,343],[858,22],[995,335],[836,215],[830,624],[460,326],[802,462],[841,708],[892,273],[784,386],[760,5],[834,602],[963,887],[327,50],[630,112],[809,328],[659,379],[349,84],[485,69],[783,408],[458,128],[976,155],[712,665],[230,28],[526,495],[425,227],[986,182],[299,162],[977,569],[829,258],[603,221],[255,28],[717,519],[924,404],[384,310],[804,552],[880,821],[895,477],[408,285],[236,53],[998,355],[688,13],[834,413],[316,245],[221,169],[951,695],[926,641],[410,266],[574,350],[707,263],[291,79],[953,497],[493,455],[824,623],[289,9],[818,749],[476,284],[529,528],[825,158],[305,152],[447,83],[304,182],[753,6],[879,332],[596,452],[815,665],[28,7],[751,657],[385,208],[985,299],[186,31],[863,176],[421,141],[743,467],[542,21],[900,324],[723,466],[193,62],[397,251],[805,167],[304,284],[622,541],[665,301],[544,17],[702,363],[746,709],[660,389],[461,398],[249,214],[555,533],[422,9],[967,32],[948,125],[566,170],[773,766],[997,54],[958,382],[163,68],[644,396],[603,2],[850,36],[433,292],[962,144],[712,466],[770,170],[992,149],[351,198],[326,127],[760,517],[892,258],[698,522],[920,679],[608,479],[724,163],[664,605],[559,17],[417,343],[822,63],[715,578],[292,16],[346,119],[826,798],[895,567],[325,21],[661,318],[614,152],[666,549],[233,30],[518,403],[766,370],[379,220],[211,62],[403,348],[208,55],[195,131],[879,241],[351,336],[990,10],[849,184],[720,49],[308,129],[439,356],[934,132],[571,210],[952,857],[845,376],[466,34],[343,300],[915,737],[487,78],[912,518],[620,418],[902,827],[887,110],[551,6],[764,14],[60,35],[230,139],[569,540],[521,244],[260,42],[943,523],[955,371],[505,20],[682,157],[582,149],[723,63],[942,461],[504,445],[933,733],[417,80],[626,16],[952,509],[313,65],[419,228],[462,125],[901,712],[759,201],[770,417],[827,672],[763,557],[571,243],[345,279],[873,857],[659,298],[845,140],[287,19],[774,484],[486,301],[648,71],[522,375],[891,618],[866,559],[460,412],[351,267],[881,252],[914,843],[465,400],[416,212],[97,21],[986,376],[137,78],[820,770],[769,274],[615,159],[858,176],[825,346],[321,49],[809,97],[244,16],[684,82],[666,379],[831,492],[805,358],[814,80],[725,142],[384,125],[383,327],[897,707],[575,445],[616,330],[913,463],[659,33],[831,653],[923,635],[618,614],[877,227],[118,68],[858,67],[779,569],[884,203],[832,711],[988,101],[673,96],[507,264],[276,177],[498,281],[763,742],[884,117],[865,312],[950,11],[549,75],[718,626],[732,108],[635,53],[687,529],[703,376],[709,394],[710,595],[821,600],[681,653],[957,265],[876,664],[781,684],[141,111],[904,490],[987,480],[698,134],[924,201],[373,255],[863,320],[709,9],[626,12],[737,512],[967,427],[447,346],[785,240],[721,209],[966,603],[735,49],[559,31],[715,520],[413,64],[623,449],[797,736],[385,93],[491,466],[615,415],[875,168],[727,27],[704,473],[518,106],[483,291],[661,342],[793,248],[585,93],[801,669],[368,303],[517,404],[759,454],[291,147],[680,675],[874,341],[690,288],[852,842],[693,60],[455,396],[459,399],[429,103],[697,210],[714,472],[208,190],[932,929],[316,27],[786,569],[719,502],[690,202],[747,281],[603,573],[470,195],[258,104],[188,74],[715,350],[959,338],[696,81],[822,204],[989,524],[654,339],[555,321],[219,38],[971,327],[581,411],[671,511],[658,46],[439,108],[251,13],[255,102],[979,264],[446,316],[748,243],[571,366],[419,403],[345,50],[972,16],[738,513],[509,359],[461,110],[929,581],[811,585],[518,322],[568,34],[900,49],[845,381],[544,53],[964,576],[949,552],[994,931],[715,462],[299,199],[113,0],[921,518],[751,630],[767,328],[812,106],[910,784],[880,713],[773,720],[971,134],[450,379],[704,146],[379,170],[746,399],[790,113],[391,2],[799,116],[845,193],[909,728],[319,62],[620,172],[528,434],[691,112],[621,588],[974,525],[982,831],[120,111],[970,76],[343,114],[633,64],[935,683],[667,326],[843,399],[973,718],[997,538],[725,282],[511,43],[192,114],[967,856],[185,87],[760,164],[435,18],[678,364],[397,352],[500,47],[949,864],[875,675],[699,583],[644,193],[759,363],[732,367],[850,460],[684,121],[778,131],[785,403],[821,7],[536,275],[577,332],[805,63],[582,75],[736,189],[396,349],[726,473],[964,659],[70,30],[894,206],[760,108],[594,182],[648,452],[931,441],[744,474],[971,868],[655,350],[725,276],[932,596],[759,515],[499,335],[941,535],[767,435],[687,531],[348,325],[389,88],[581,419],[742,665],[495,188],[890,177],[927,674],[711,217],[361,68],[487,407],[525,437],[975,42],[599,122],[909,632],[934,16],[880,165],[178,150],[826,401],[433,119],[174,63],[998,682],[664,571],[997,648],[593,421],[58,53],[292,146],[982,82],[734,682],[357,43],[523,275],[330,282],[797,599],[918,27],[478,50],[929,117],[739,534],[268,148],[439,331],[635,424],[901,465],[748,583],[773,385],[742,598],[554,227],[190,57],[523,10],[542,390],[311,204],[952,72],[499,433],[549,359],[932,502],[743,514],[560,438],[395,223],[594,248],[860,830],[291,179],[575,273],[994,978],[721,325],[683,548],[891,445],[204,135],[979,534],[738,244],[656,572],[534,302],[979,849],[932,829],[763,751],[284,152],[137,104],[415,213],[466,284],[180,21],[959,666],[856,513],[960,888],[733,412],[618,290],[589,463],[531,475],[927,845],[849,179],[735,504],[774,217],[992,580],[144,66],[466,307],[446,324],[674,133],[736,11],[726,373],[608,565],[462,229],[726,351],[382,212],[150,31],[317,298],[405,211],[675,55],[563,342],[501,47],[765,681],[937,129],[965,186],[856,88],[837,326],[558,355],[724,149],[311,104],[481,182],[975,688],[757,640],[834,535],[395,270],[311,5],[717,149],[858,189],[457,146],[647,505],[478,279],[832,308],[286,139],[474,296],[608,482],[366,186],[983,189],[949,256],[919,344],[948,794],[720,251],[755,364],[971,249],[840,512],[859,815],[574,129],[758,281],[907,145],[328,92],[909,583],[990,688],[661,100],[777,188],[277,150],[839,192],[532,507],[834,356],[170,13],[514,211],[277,233],[885,630],[371,245],[985,383],[769,281],[886,135],[896,212],[372,203],[767,312],[713,676],[960,60],[521,50],[627,243],[578,512],[575,326],[47,0],[912,851],[439,192],[777,220],[948,751],[265,235],[963,102],[743,664],[514,218],[994,24],[834,441],[790,300],[679,141],[71,14],[965,87],[801,657],[624,596],[288,86],[533,106],[447,132],[881,223],[889,233],[517,293],[488,89],[284,14],[409,54],[542,221],[293,225],[570,106],[902,641],[937,642],[643,358],[467,73],[741,286],[341,154],[768,138],[440,286],[119,6],[967,451],[917,654],[638,259],[982,76],[501,186],[638,450],[558,164],[932,743],[911,733],[292,74],[962,417],[294,85],[136,103],[395,167],[197,167],[985,682],[865,494],[879,407],[708,419],[692,196],[790,144],[634,500],[718,214],[600,92],[320,213],[286,278],[591,282],[720,565],[606,499],[998,49],[992,316],[719,596],[930,36],[847,583],[256,46],[474,185],[583,114],[856,670],[291,100],[487,235],[739,492],[360,358],[625,239],[792,449],[771,701],[867,598],[866,501],[293,265],[863,115],[783,358],[932,396],[524,282],[125,108],[755,474],[894,566],[581,32],[312,193],[799,670],[967,717],[621,36],[800,428],[599,219],[652,384],[607,136],[825,610],[367,268],[533,7],[772,275],[747,577],[887,124],[888,127],[922,739],[909,744],[498,237],[944,192],[653,28],[587,322],[361,206],[555,313],[533,294],[21,11],[381,340],[710,65],[696,584],[783,558],[235,82],[874,380],[656,125],[972,199],[685,400],[947,450],[949,617],[601,267],[828,278],[459,78],[269,122],[235,75],[563,8],[478,408],[803,8],[868,741],[813,115],[789,370],[837,453],[856,130],[501,356],[687,74],[553,114],[282,174],[123,69],[995,208],[823,241],[150,58],[678,296],[755,524],[833,325],[761,539],[807,600],[613,435],[936,707],[622,132],[802,645],[737,7],[998,142],[283,102],[941,635],[351,287],[922,810],[604,48],[573,55],[340,85],[763,359],[773,44],[270,186],[957,181],[956,764],[895,683],[742,434],[832,30],[623,280],[680,235],[420,298],[814,239],[870,755],[838,464],[234,227],[928,617],[966,4],[835,680],[499,383],[683,507],[895,878],[778,154],[629,596],[875,20],[998,427],[985,318],[590,81],[727,74],[430,212],[867,367],[783,400],[978,526],[836,339],[896,330],[993,896],[152,42],[884,497],[745,134],[835,808],[977,497],[477,464],[847,633],[588,51],[546,384],[807,528],[938,922],[800,75],[915,110],[809,797],[682,622],[747,590],[856,732],[376,45],[719,631],[744,674],[238,0],[424,330],[828,119],[963,754],[686,19],[758,609],[666,366],[761,243],[724,231],[817,801],[971,408],[595,429],[684,67],[110,45],[989,855],[632,178],[858,696],[979,286],[723,51],[280,260],[913,97],[760,601],[468,227],[189,56],[996,187],[664,350],[448,80],[290,204],[935,352],[437,94],[893,498],[451,85],[364,174],[271,198],[917,638],[959,46],[498,239],[761,310],[751,587],[959,284],[518,217],[225,49],[667,133],[576,107],[855,450],[395,393],[676,532],[961,1],[533,210],[861,483],[800,586],[461,52],[908,431],[532,384],[697,188],[674,184],[486,6],[815,22],[782,424],[994,772],[406,194],[441,392],[904,255],[998,867],[966,370],[268,169],[347,153],[709,645],[889,583],[248,102],[752,581],[588,430],[189,102],[760,215],[296,291],[931,87],[619,561],[823,33],[652,535],[340,281],[263,40],[821,471],[865,78],[140,4],[930,258],[733,201],[755,420],[579,198],[895,801],[757,192],[511,215],[481,452],[429,51],[919,313],[580,49]]
# #     print(s.canFinish(1000, prerequisites))
#
#     prerequisites = [[1, 0]]
#     print("can finish: ")
#     print(s.canFinish(3, prerequisites))
#     print("course order: ")
#     print(s.findOrder(3, prerequisites))
#
#     rebuiltTree = s.buildTree(['F', 'B', 'A', 'D', 'C', 'E', 'G', 'I', 'H'],
#                               ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
#     print(rebuiltTree.val)
#
#     bstTree = s.sortedArrayToBST([-10, -3, 0, 5, 9])
#     print(bstTree.val)
#     myBSTList = ListNode(-10)
#     myBSTList.insertNode(-3)
#     myBSTList.insertNode(0)
#     myBSTList.insertNode(5)
#     myBSTList.insertNode(9)
#     bstTree2 = s.sortedListToBST(myBSTList)
#     print(bstTree2.val)
#
#     print(s.maximumProduct([-1, -2, -3]))
#
#     print("Max Profit One transaction:")
#     print(s.maxProfit([7,1,5,3,6,4]))
#     print("Max Profit Two transaction:")
#     print(s.maxProfit3([3,3,5,0,0,3,1,4]))
#
#     print(3//2)
#
#     print(s.calculate2("2 + 3 *2/5*7+10/5*2"))
#
#     myTree = TreeNode(5)
#     myTree.left = TreeNode(4)
#     myTree.right = TreeNode(8)
#     myTree.left.left = TreeNode(11)
#     myTree.left.left.left = TreeNode(7)
#     myTree.left.left.right = TreeNode(2)
#     myTree.right.left = TreeNode(13)
#     myTree.right.right = TreeNode(4)
#     myTree.right.right.left = TreeNode(5)
#     myTree.right.right.right = TreeNode(1)
#     print(s.hasPathSum(myTree, 22))
#     print(s.pathSum(myTree, 22))
#
#     tree2Flat = TreeNode(1)
#     tree2Flat.left = TreeNode(2)
#     tree2Flat.right = TreeNode(5)
#     tree2Flat.left.left = TreeNode(3)
#     tree2Flat.left.right = TreeNode(4)
#     tree2Flat.right.right = TreeNode(6)
#     s.flatten(tree2Flat)
#     print(tree2Flat.val)
#
#     mybtTree = TreeNode(3)
#     mybtTree.left = TreeNode(9)
#     mybtTree.right = TreeNode(20)
#     mybtTree.right.left = TreeNode(15)
#     mybtTree.right.right = TreeNode(7)
#     print(s.levelOrder(mybtTree))
#
#     grid = [[1,3,1],
#             [1,5,1],
#             [4,2,1]]
#     print(s.minPathSum(grid))
#
#     triangle = [[2],
#                 [3,4],
#                 [6,5,7],
#                 [4,1,8,3]]
#     print(s.minimumTotal(triangle))

main()
